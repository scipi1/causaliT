"""
Gradient Routing: Selective parameter updates for structure vs reconstruction.

In SVFA (Structure-Value Factorized Attention), the model naturally separates
into structural parameters (controlling which variables attend to which) and
reconstruction parameters (controlling how information flows and predictions
are made). This module classifies parameters into these groups and provides
dual-optimizer support so that:

- HSIC (+ group L1, score sparsity) gradients only update structural parameters
- Reconstruction loss gradients only update reconstruction parameters

This prevents the gradient conflict where HSIC noise drowns out the
reconstruction signal and vice versa.

Parameter Classification:
    Structural (theta_S):
        - Structure embeddings (nn_embedding for variable IDs)
        - Free query embedding (decoupled X query, feeds Q only)
        - Q, K projections in all attention layers
        - Attention internal params (log_gain, log_tau, etc.)
        - Structure-path layer norms (norm1_struct, norm2_struct)
    
    Reconstruction (theta_R):
        - Value embeddings (linear for values)
        - V projection in all attention layers
        - Output projection in attention layers
        - FF layers (linear1, linear2)
        - MLP head / forecaster
        - Value-path layer norms (norm1, norm2, norm3, norm_layer)
        - Noise parameters (sigma_A, sigma_R) for noise-aware models

References:
    - PCGrad (Yu et al., NeurIPS 2020): Gradient surgery for multi-task learning
    - DARTS (Liu et al., ICLR 2019): Bilevel optimization for architecture search
    - GradNorm (Chen et al., ICML 2018): Adaptive gradient balancing

Usage:
    from causaliT.training.gradient_routing import classify_parameters
    
    structural, reconstruction = classify_parameters(model)
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER NAME PATTERNS
# =============================================================================

# Patterns that identify structural parameters (Q, K, structure embeddings,
# attention internals, structure norms).  Order doesn't matter - we check
# with substring matching.
STRUCTURAL_PATTERNS = [
    # Q, K projections (structure determines attention pattern)
    "query_projection",
    "key_projection",
    # Structural value & output projections (SVFA dual-residual /
    # ``AttentionLayer(dual_value=True)``).  These produce the structural
    # residual added to X_struct, so they belong to the structural group
    # and are driven by HSIC / score-sparsity, not the reconstruction loss.
    # The patterns must keep the trailing "_struct" so they are NOT
    # confused with the reconstruction "value_projection" / "out_projection".
    "value_projection_struct",
    "out_projection_struct",
    # Structure embeddings in SVFA (nn_embedding for variable IDs)
    # In SVFAEmbedding, structure modules have "structure" in the path.
    # In the standard DataSetEmbedding with role="structure", the module
    # label is "variable" and uses nn.Embedding.
    "structure_modules",
    # Orthogonal structural key embeddings (AttentionSelectorLayer,
    # struct_embedding_type="orthogonal_learnable"/"orthogonal_fixed").
    # Both schemes are stored under the SAME attribute names
    # (``orth_embed_S`` / ``orth_embed_X``) in the model, so one pair of
    # patterns covers both:
    #   * "orthogonal_learnable" -> OrthogonalMaskEmbedding, whose learnable
    #     ``value_embedding`` (nn.Linear, freeze=False) builds the structural
    #     keys that feed the gate score log_alpha = QK^T -> STRUCTURAL.
    #   * "orthogonal_fixed"     -> FixedOrthonormalEmbedding, a frozen
    #     buffer-only frame (no trainable params) -> matches nothing today,
    #     but the rule stays consistent and future-proof.
    # Without these, the learnable orthogonal key embeddings were wrongly
    # routed to the reconstruction group (frozen in the structure phase,
    # trained by HSIC/L0 in the reconstruct phase -- the exact inverse of the
    # intended gradient routing).
    "orth_embed_S",
    "orth_embed_X",

    # Free query embedding (AttentionSelectorLayer, free_query_embedding=True).
    # This is the decoupled X *query* identity lookup table; it feeds only the
    # attention Query, so it is a structural parameter driven by HSIC /
    # score-sparsity — NOT the reconstruction loss.
    # Prefix (NOT the full name) so that BOTH tables are matched:
    #   * ``query_embed_X`` — split mode (X children only)
    #   * ``query_embed_S`` — homogeneous_nodes=True, where the S variables are
    #     children too and therefore own a free query table as well.
    # Verified: no reconstruction parameter name contains "query_embed".
    "query_embed",

    # CommutatorSelfAttention direction generator (direction_mode="skew_query").
    # The bias-free projections ``direction_proj_a`` / ``direction_proj_b`` build
    # the learnable so(d) commutator Ω = W_a W_bᵀ − W_b W_aᵀ that resolves EDGE
    # DIRECTION; they are driven by HSIC / score-sparsity → STRUCTURAL.  The
    # substring "direction_proj" does not collide with the reconstruction
    # "value_projection" / "out_projection" patterns.
    "direction_proj",
    # Learnable per-node query-norm budget (M_i = exp(log_scale_i)).  It scales
    # the structural query direction and is charged the over-spend penalty on
    # the STRUCTURAL loss, so it belongs to the structural group.
    "query_norm_log_scale",
    # Attention internal parameters (gain, temperature, Gumbel tau, etc.)

    "inner_attention.log_gain",
    "inner_attention.log_tau",
    "inner_attention.temperature",
    # Structure-path layer norms (only exist in SVFA decoders)
    "norm1_struct",
    "norm2_struct",
]

# Patterns that identify reconstruction parameters.  Anything not matched
# by STRUCTURAL_PATTERNS falls here by default, but we list explicit
# patterns for clarity and for the "mixed" detection warning.
RECONSTRUCTION_PATTERNS = [
    # V projection and output projection (value information flow)
    "value_projection",
    "out_projection",
    # Value embeddings in SVFA
    "value_modules",
    # FF layers (transform attended values)
    "linear1",
    "linear2",
    # MLP head / forecaster (final prediction)
    "forecaster",
    "output_head",
    # Value-path layer norms
    # (norm1, norm2, norm3 that are NOT norm1_struct/norm2_struct)
    # Noise parameters (noise-aware models)
    "ambient_noise",
    "sigma_A",
    "sigma_R",
    # Final normalization
    "norm_layer",
]


def _is_structural_param(name: str) -> bool:
    """Check if a parameter name corresponds to a structural parameter."""
    for pattern in STRUCTURAL_PATTERNS:
        if pattern in name:
            return True
    return False


def classify_parameters(
    model: nn.Module,
    verbose: bool = False,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Classify model parameters into structural and reconstruction groups.
    
    Uses name-based pattern matching on the parameter names. Parameters
    matching STRUCTURAL_PATTERNS are classified as structural; all others
    are classified as reconstruction.
    
    Args:
        model:   The inner model (e.g., SingleCausalLayer or
                 NoiseAwareSingleCausalLayer). NOT the Lightning wrapper.
        verbose: If True, print classification details.
    
    Returns:
        Tuple of (structural_params, reconstruction_params), each a list
        of nn.Parameter objects (with requires_grad=True only).
    """
    structural_params = []
    reconstruction_params = []
    
    structural_names = []
    reconstruction_names = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if _is_structural_param(name):
            structural_params.append(param)
            structural_names.append(name)
        else:
            reconstruction_params.append(param)
            reconstruction_names.append(name)
    
    n_struct = sum(p.numel() for p in structural_params)
    n_recon = sum(p.numel() for p in reconstruction_params)
    n_total = n_struct + n_recon
    
    if verbose or logger.isEnabledFor(logging.DEBUG):
        msg = (
            f"Gradient routing parameter classification:\n"
            f"  Structural:      {len(structural_params):3d} tensors, "
            f"{n_struct:,d} params ({100*n_struct/max(n_total,1):.1f}%)\n"
            f"  Reconstruction:  {len(reconstruction_params):3d} tensors, "
            f"{n_recon:,d} params ({100*n_recon/max(n_total,1):.1f}%)"
        )
        if verbose:
            print(msg)
            print("\n  Structural parameters:")
            for n in structural_names:
                print(f"    theta_S: {n}")
            print("\n  Reconstruction parameters:")
            for n in reconstruction_names:
                print(f"    theta_R: {n}")
        logger.debug(msg)
    
    if len(structural_params) == 0:
        logger.warning(
            "Gradient routing: no structural parameters found! "
            "Check that SVFA factorization is enabled."
        )
    if len(reconstruction_params) == 0:
        logger.warning(
            "Gradient routing: no reconstruction parameters found!"
        )
    
    return structural_params, reconstruction_params


def log_parameter_classification(
    model: nn.Module,
    structural_params: List[torch.nn.Parameter],
    reconstruction_params: List[torch.nn.Parameter],
) -> Dict[str, any]:
    """
    Create a summary dict of parameter classification for logging.
    
    Returns:
        Dict with classification summary statistics.
    """
    n_struct = sum(p.numel() for p in structural_params)
    n_recon = sum(p.numel() for p in reconstruction_params)
    
    # Map param id -> group for named lookup
    struct_ids = {id(p) for p in structural_params}
    recon_ids = {id(p) for p in reconstruction_params}
    
    struct_names = []
    recon_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in struct_ids:
            struct_names.append(name)
        elif id(param) in recon_ids:
            recon_names.append(name)
    
    return {
        "n_structural_tensors": len(structural_params),
        "n_reconstruction_tensors": len(reconstruction_params),
        "n_structural_params": n_struct,
        "n_reconstruction_params": n_recon,
        "structural_param_names": struct_names,
        "reconstruction_param_names": recon_names,
    }
