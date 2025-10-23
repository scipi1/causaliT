import torch
import threading
from typing import Dict, List, Optional
from collections import defaultdict


class AttentionEntropyRegistry:
    """
    Thread-safe global registry for collecting attention entropy values during training.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._entropy_values = defaultdict(list)
        self._enabled = True
    
    def register_entropy(self, layer_name: str, entropy: torch.Tensor):
        """
        Register entropy values for a specific attention layer.
        
        Args:
            layer_name: Name identifier for the attention layer
            entropy: Entropy tensor with shape (B, H, L) for multi-head or (B, L) for single-head
        """
        if not self._enabled:
            return
            
        with self._lock:
            # Detach and move to CPU to avoid memory issues
            entropy_cpu = entropy.detach().cpu()
            self._entropy_values[layer_name].append(entropy_cpu)
    
    def get_aggregated_entropy(self) -> Dict[str, torch.Tensor]:
        """
        Get aggregated entropy statistics across all registered values.
        
        Returns:
            Dictionary with entropy statistics
        """
        with self._lock:
            aggregated = {}
            
            for layer_name, entropy_list in self._entropy_values.items():
                
                if not entropy_list:
                    continue
                
                
                # Stack all entropy tensors for this layer
                stacked_entropy = torch.cat(entropy_list, dim=0)  # (Total_B, H, L) or (Total_B, L)
                
                if stacked_entropy.dim() == 3:  # Multi-head case (B, H, L)
                    # Overall mean entropy
                    aggregated[f"{layer_name}_entropy_mean"] = stacked_entropy.mean()
                    
                    # Per-head entropy
                    n_heads = stacked_entropy.size(1)
                    for head_idx in range(n_heads):
                        head_entropy = stacked_entropy[:, head_idx, :].mean()
                        aggregated[f"{layer_name}_entropy_head_{head_idx}"] = head_entropy
                        
                elif stacked_entropy.dim() == 2:  # Single-head case (B, L)
                    aggregated[f"{layer_name}_entropy_mean"] = stacked_entropy.mean()
            
            return aggregated
    
    def clear(self):
        """Clear all registered entropy values."""
        with self._lock:
            self._entropy_values.clear()
    
    def enable(self):
        """Enable entropy collection."""
        with self._lock:
            self._enabled = True
    
    def disable(self):
        """Disable entropy collection."""
        with self._lock:
            self._enabled = False


# Global registry instance
_global_entropy_registry = AttentionEntropyRegistry()


def get_entropy_registry() -> AttentionEntropyRegistry:
    """Get the global entropy registry instance."""
    return _global_entropy_registry


def calculate_attention_entropy(att_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate entropy of attention weights.
    
    Args:
        att_weights: Attention weights tensor
                    - Multi-head: (B, H, L, S) 
                    - Single-head: (B, L, S)
        eps: Small value to avoid log(0)
    
    Returns:
        Entropy tensor:
        - Multi-head: (B, H, L) - entropy for each query position in each head
        - Single-head: (B, L) - entropy for each query position
    """
    # Clamp to avoid log(0)
    att_clamped = torch.clamp(att_weights, min=eps)
    
    # Calculate entropy: -sum(p * log(p)) along the key dimension (last dimension)
    log_att = torch.log(att_clamped)
    entropy = -torch.sum(att_weights * log_att, dim=-1)
    
    # Handle NaN values that might arise from 0 * log(0)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    return entropy


def register_attention_entropy(layer_name: str, att_weights: torch.Tensor):
    """
    Calculate and register attention entropy for a given layer.
    
    Args:
        layer_name: Name identifier for the attention layer
        att_weights: Attention weights tensor
    """
    entropy = calculate_attention_entropy(att_weights)
    get_entropy_registry().register_entropy(layer_name, entropy)
