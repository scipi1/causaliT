"""
In-context mask generation functions for the dyconex dataset.

These functions generate attention masks computed online (per-batch) from data features,
as opposed to static hard masks loaded from files.

Masks:
1. Causal Order Mask: Prevents tokens from attending to future tokens based on "order" feature
2. Category Cross-Attention Mask: Allows X tokens to attend only to S tokens of same category

The masks are compatible with the model's hard_mask parameter in attention layers.
"""

import torch
from typing import Dict, List, Optional, Union


def build_causal_order_mask(
    query_order: torch.Tensor,
    key_order: torch.Tensor,
) -> torch.Tensor:
    """
    Build causal mask based on order features.
    
    Prevents query tokens from attending to key tokens with higher order values.
    A token at query position i can only attend to key positions j where order[j] <= order[i].
    
    Args:
        query_order: Order values for query sequence, shape (B, L_q) or (B, L_q, 1)
        key_order: Order values for key sequence, shape (B, L_k) or (B, L_k, 1)
        
    Returns:
        mask: Binary mask of shape (B, L_q, L_k), where 1 = attention allowed, 0 = blocked
        
    Example:
        If query_order = [1, 2, 3] and key_order = [1, 2, 3]:
        mask = [[1, 0, 0],   # order 1 can see order 1
                [1, 1, 0],   # order 2 can see orders 1, 2
                [1, 1, 1]]   # order 3 can see orders 1, 2, 3
    """
    # Ensure 2D tensors (B, L)
    if query_order.dim() == 3:
        query_order = query_order.squeeze(-1)
    if key_order.dim() == 3:
        key_order = key_order.squeeze(-1)
    
    # Handle NaN values by replacing with a large number (will be blocked)
    query_order = torch.nan_to_num(query_order, nan=float('inf'))
    key_order = torch.nan_to_num(key_order, nan=float('inf'))
    
    # Build mask: query_order[i] >= key_order[j] means key at j is allowed
    # Shape: (B, L_q, 1) vs (B, 1, L_k) -> (B, L_q, L_k)
    q_expanded = query_order.unsqueeze(-1)  # (B, L_q, 1)
    k_expanded = key_order.unsqueeze(-2)     # (B, 1, L_k)
    
    # 1 where key_order <= query_order (past or current), 0 for future
    mask = (k_expanded <= q_expanded).float()
    
    return mask


def build_category_cross_mask(
    query_features: torch.Tensor,
    key_features: torch.Tensor,
    category_indices: List[int],
) -> torch.Tensor:
    """
    Build category-based cross-attention mask.
    
    Allows query tokens to attend only to key tokens that share the same category.
    Category is defined by a set of feature indices (e.g., process, occurrence, step).
    
    Args:
        query_features: Full feature tensor for query sequence, shape (B, L_q, D)
        key_features: Full feature tensor for key sequence, shape (B, L_k, D)
        category_indices: List of feature indices that define the category
                         e.g., [1, 2, 3] for process, occurrence, step
        
    Returns:
        mask: Binary mask of shape (B, L_q, L_k), where 1 = attention allowed (same category)
        
    Example:
        For category_indices = [process, occurrence, step]:
        - X token at (process=1, occurrence=2, step=1) can ONLY attend to
        - S tokens at (process=1, occurrence=2, step=1)
    """
    B, L_q, _ = query_features.shape
    _, L_k, _ = key_features.shape
    
    # Extract category features
    # Shape: (B, L_q, num_category_features)
    query_category = query_features[:, :, category_indices]
    # Shape: (B, L_k, num_category_features)
    key_category = key_features[:, :, category_indices]
    
    # Handle NaN values - treat NaN as a special category that matches nothing
    # Replace NaN with a unique large negative value that won't match anything
    query_category = torch.nan_to_num(query_category, nan=-9999.0)
    key_category = torch.nan_to_num(key_category, nan=-9998.0)  # Different value so NaN != NaN
    
    # Compare categories: all features must match
    # Expand for broadcasting: (B, L_q, 1, C) vs (B, 1, L_k, C)
    query_expanded = query_category.unsqueeze(2)  # (B, L_q, 1, C)
    key_expanded = key_category.unsqueeze(1)       # (B, 1, L_k, C)
    
    # Check if all category features match
    # (B, L_q, L_k, C) -> (B, L_q, L_k)
    matches = (query_expanded == key_expanded).all(dim=-1).float()
    
    return matches


def build_dyconex_in_context_masks(
    S: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Build all in-context masks for the dyconex dataset.
    
    This master function generates masks for all four attention types:
    - dec1_cross: X (query) attending to S (key) - causal + category masks combined
    - dec1_self: X self-attention - causal mask only
    - dec2_cross: Y (query) attending to X (key) - causal mask only
    - dec2_self: Y self-attention - causal mask only
    
    Args:
        S: Source tensor, shape (B, S_len, D_s)
        X: Intermediate tensor, shape (B, X_len, D_x)  
        Y: Target tensor, shape (B, Y_len, D_y)
        config: Configuration dictionary with mask settings. Expected structure:
            {
                'dec1_cross': {
                    'causal': True,
                    'category': True,
                    'order_idx_q': 6,      # X order feature index
                    'order_idx_k': 6,      # S order feature index
                    'category_idx': [1, 2, 3]  # process, occurrence, step indices
                },
                'dec1_self': {
                    'causal': True,
                    'order_idx': 6
                },
                'dec2_cross': {
                    'causal': True,
                    'order_idx_q': 1,      # Y position feature index
                    'order_idx_k': 6       # X order feature index
                },
                'dec2_self': {
                    'causal': True,
                    'order_idx': 1         # Y position feature index
                }
            }
            
    Returns:
        Dict with keys: 'dec1_cross', 'dec1_self', 'dec2_cross', 'dec2_self'
        Each value is a mask tensor of appropriate shape, values in [0, 1]
    """
    masks = {}
    
    # =========================================================================
    # Decoder 1 Cross-Attention: X queries S
    # =========================================================================
    dec1_cross_config = config.get('dec1_cross', {})
    
    if dec1_cross_config:
        dec1_cross_mask = None
        
        # Causal order mask
        if dec1_cross_config.get('causal', False):
            order_idx_q = dec1_cross_config['order_idx_q']
            order_idx_k = dec1_cross_config['order_idx_k']
            
            x_order = X[:, :, order_idx_q]  # (B, X_len)
            s_order = S[:, :, order_idx_k]  # (B, S_len)
            
            causal_mask = build_causal_order_mask(x_order, s_order)
            dec1_cross_mask = causal_mask
        
        # Category mask (specific to dec1_cross)
        if dec1_cross_config.get('category', False):
            category_indices = dec1_cross_config['category_idx']
            
            category_mask = build_category_cross_mask(X, S, category_indices)
            
            # Combine with causal mask (element-wise AND)
            if dec1_cross_mask is not None:
                dec1_cross_mask = dec1_cross_mask * category_mask
            else:
                dec1_cross_mask = category_mask
        
        if dec1_cross_mask is not None:
            masks['dec1_cross'] = dec1_cross_mask
    
    # =========================================================================
    # Decoder 1 Self-Attention: X self-attention
    # =========================================================================
    dec1_self_config = config.get('dec1_self', {})
    
    if dec1_self_config and dec1_self_config.get('causal', False):
        order_idx = dec1_self_config['order_idx']
        
        x_order = X[:, :, order_idx]  # (B, X_len)
        
        masks['dec1_self'] = build_causal_order_mask(x_order, x_order)
    
    # =========================================================================
    # Decoder 2 Cross-Attention: Y queries X
    # =========================================================================
    dec2_cross_config = config.get('dec2_cross', {})
    
    if dec2_cross_config and dec2_cross_config.get('causal', False):
        order_idx_q = dec2_cross_config['order_idx_q']
        order_idx_k = dec2_cross_config['order_idx_k']
        
        y_order = Y[:, :, order_idx_q]  # (B, Y_len) - using position
        x_order = X[:, :, order_idx_k]  # (B, X_len) - using order
        
        masks['dec2_cross'] = build_causal_order_mask(y_order, x_order)
    
    # =========================================================================
    # Decoder 2 Self-Attention: Y self-attention
    # =========================================================================
    dec2_self_config = config.get('dec2_self', {})
    
    if dec2_self_config and dec2_self_config.get('causal', False):
        order_idx = dec2_self_config['order_idx']
        
        y_order = Y[:, :, order_idx]  # (B, Y_len)
        
        masks['dec2_self'] = build_causal_order_mask(y_order, y_order)
    
    return masks


def merge_masks(
    static_masks: Optional[Dict[str, torch.Tensor]],
    in_context_masks: Optional[Dict[str, torch.Tensor]],
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Merge static hard masks with in-context computed masks.
    
    When both masks are present for the same attention layer, they are combined
    using element-wise multiplication (logical AND).
    
    Args:
        static_masks: Dictionary of static masks (from file), or None
        in_context_masks: Dictionary of in-context masks (computed from data), or None
        
    Returns:
        Merged dictionary of masks, or None if both inputs are None
    """
    if static_masks is None and in_context_masks is None:
        return None
    
    if static_masks is None:
        return in_context_masks
    
    if in_context_masks is None:
        return static_masks
    
    # Merge: combine masks where both exist, keep single masks otherwise
    merged = {}
    all_keys = set(static_masks.keys()) | set(in_context_masks.keys())
    
    for key in all_keys:
        static = static_masks.get(key)
        dynamic = in_context_masks.get(key)
        
        if static is not None and dynamic is not None:
            # Both exist: element-wise AND
            # Handle potential shape mismatches (static may lack batch dim)
            if static.dim() < dynamic.dim():
                static = static.unsqueeze(0).expand_as(dynamic)
            merged[key] = static * dynamic
        elif static is not None:
            merged[key] = static
        else:
            merged[key] = dynamic
    
    return merged
