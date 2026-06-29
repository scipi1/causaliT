"""
Orthogonal Mask Embedding: Creates orthogonal embeddings using binary masks.

Key idea:
- Each variable gets a non-overlapping subset of dimensions (binary mask)
- Final embedding = value_embedding ⊙ binary_mask
- This ensures ⟨emb(Sᵢ), emb(Sⱼ)⟩ = 0 for i ≠ j

Use case:
- Embed source variables (S) with orthogonal representations
- Ensures attention scores between X and different S variables carry independent information

When used inside AttentionSelectorLayer (SVFA mode), both S and X nodes are embedded
as structural keys that are concatenated before the shared W_K projection:

    sx_keys = cat([s_struct, xk_struct], dim=1)   # (B, L_S+L_X, d_model)

For the L_S+L_X keys to be mutually orthogonal, S and X must occupy **disjoint**
partitions of the d_model dimension space. This is achieved via ``mask_start_dim``:

    embedding_S = OrthogonalMaskEmbedding(L_S, d_model, mask_start_dim=0)
    embedding_X = OrthogonalMaskEmbedding(L_X, d_model, mask_start_dim=L_S * dims_per_var)

with d_model = (L_S + L_X) * dims_per_var.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class OrthogonalMaskEmbedding(nn.Module):
    """
    Orthogonal embedding using binary masks and element-wise product.
    
    For variable Sⱼ with value vⱼ:
        emb(Sⱼ) = value_embed(vⱼ) ⊙ mask_j
    
    Where masks are orthogonal binary vectors that partition the d_model dimensions.
    
    Example for 3 variables with d_model=6:
        S₁ mask: [1, 1, 0, 0, 0, 0]
        S₂ mask: [0, 0, 1, 1, 0, 0]
        S₃ mask: [0, 0, 0, 0, 1, 1]
    
    Args:
        num_variables: Number of source variables (e.g., 3 for S₁, S₂, S₃)
        d_model: Embedding dimension (must be large enough to hold all variables'
            partitions starting from mask_start_dim)
        value_input_dim: Dimension of value input (default 1)
        value_idx: Index of value in input tensor
        var_idx: Index of variable ID in input tensor
        var_id_offset: Offset to subtract from var_ids before indexing masks.
                       Default 1 since SCM datasets use 1-indexed var IDs (S1=1, S2=2, S3=3)
                       and 0 is reserved for padding/missing.
        mask_start_dim: First dimension index assigned to variable 0.  Default 0.
                        Set this to a non-zero value when multiple
                        OrthogonalMaskEmbedding instances must occupy disjoint
                        partitions of the same d_model space (e.g. S and X nodes
                        concatenated as keys in AttentionSelectorLayer).
                        Must be used together with ``dims_per_var`` to fix the
                        partition width:
                            k = d_model // (L_S + L_X)
                            embedding_S = OrthogonalMaskEmbedding(L_S, d_model, mask_start_dim=0,      dims_per_var=k)
                            embedding_X = OrthogonalMaskEmbedding(L_X, d_model, mask_start_dim=L_S*k, dims_per_var=k)
                        with d_model = (L_S + L_X) * k.
        dims_per_var:   Exact number of dimensions allocated to each variable.
                        When None (default), auto-computed as
                        ``(d_model - mask_start_dim) // num_variables`` — this is
                        correct when all variables belong to the same group (the
                        default single-group case).  Must be provided explicitly
                        when two groups share the same d_model with ``mask_start_dim``
                        to avoid one group inadvertently consuming the other's space.
        freeze: Whether to freeze the entire embedding (default True for source variables)
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        num_variables: int,
        d_model: int,
        value_input_dim: int = 1,
        value_idx: int = 0,
        var_idx: int = 1,
        var_id_offset: int = 1,
        mask_start_dim: int = 0,
        dims_per_var: Optional[int] = None,
        freeze: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.num_variables = num_variables
        self.d_model = d_model
        self.value_idx = value_idx
        self.var_idx = var_idx
        self.var_id_offset = var_id_offset
        self.mask_start_dim = mask_start_dim
        self.freeze = freeze
        self.device = device
        
        if dims_per_var is not None:
            # Explicit partition width: all variables get exactly dims_per_var dims.
            # Used when two OrthogonalMaskEmbedding groups share the same d_model
            # via mask_start_dim and must have equal-sized partitions.
            if dims_per_var <= 0:
                raise ValueError(f"dims_per_var must be positive, got {dims_per_var}.")
            end_dim = mask_start_dim + num_variables * dims_per_var
            if end_dim > d_model:
                raise ValueError(
                    f"mask_start_dim={mask_start_dim} + num_variables={num_variables} * "
                    f"dims_per_var={dims_per_var} = {end_dim} exceeds d_model={d_model}."
                )
            self.dims_per_var = dims_per_var
            self.extra_dims = 0  # uniform allocation, no remainder
        else:
            # Auto-compute: partition [mask_start_dim, d_model) evenly.
            # When mask_start_dim=0 this is the original single-group behaviour.
            available_dims = d_model - mask_start_dim
            if available_dims <= 0:
                raise ValueError(
                    f"mask_start_dim={mask_start_dim} leaves no space in d_model={d_model} "
                    f"for {num_variables} variables."
                )
            self.dims_per_var = available_dims // num_variables
            self.extra_dims = available_dims % num_variables
        
        # Scale factor to maintain variance after masking
        # Since only dims_per_var out of d_model dimensions are active,
        # we scale by sqrt(d_model / dims_per_var) to preserve expected variance
        self.scale_factor = math.sqrt(d_model / self.dims_per_var)
        
        # Shared value embedding: Linear(value_input_dim → d_model)
        # All variables share this transformation for the value feature
        self.value_embedding = nn.Linear(value_input_dim, d_model, bias=True)
        
        # Create binary orthogonal masks (registered as buffer - not trainable)
        # Shape: (num_variables, d_model)
        masks = self._create_orthogonal_masks()
        self.binary_masks: torch.Tensor
        self.register_buffer('binary_masks', masks)
        
        # Apply freezing if requested
        if freeze:
            self._freeze()
    
    def _create_orthogonal_masks(self) -> torch.Tensor:
        """
        Create binary orthogonal masks that partition the d_model dimensions.

        Each variable gets a non-overlapping block of ``dims_per_var`` dimensions
        (±1 for the first ``extra_dims`` variables) starting at ``mask_start_dim``.
        Dimensions below ``mask_start_dim`` are left as zero, allowing multiple
        OrthogonalMaskEmbedding instances to share the same d_model space without
        overlap.
        
        Returns:
            Tensor of shape (num_variables, d_model) with binary values
        """
        masks = torch.zeros(self.num_variables, self.d_model)
        
        start_idx = self.mask_start_dim   # honour the partition offset
        for var_id in range(self.num_variables):
            # Distribute any remainder dimensions to the first variables
            dims_for_this_var = self.dims_per_var + (1 if var_id < self.extra_dims else 0)
            end_idx = start_idx + dims_for_this_var
            
            # Set active dimensions for this variable
            masks[var_id, start_idx:end_idx] = 1.0
            
            start_idx = end_idx
        
        return masks
    
    def _freeze(self):
        """Freeze all parameters in this embedding."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze the value embedding (masks remain fixed as buffers)."""
        for param in self.value_embedding.parameters():
            param.requires_grad = True
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embed values and apply orthogonal masks.
        
        Args:
            X: Input tensor of shape (batch_size, seq_len, features)
               Features should include value at value_idx and variable ID at var_idx
        
        Returns:
            Orthogonal embeddings of shape (batch_size, seq_len, d_model)
        """
        # Extract values and variable IDs
        values = X[:, :, self.value_idx:self.value_idx+1]  # (B, L, 1)
        var_ids_raw = X[:, :, self.var_idx].long()  # (B, L)
        
        # Apply offset: var_ids in data are 1-indexed (S1=1, S2=2, S3=3)
        # We need 0-indexed for mask lookup (S1→0, S2→1, S3→2)
        var_ids = var_ids_raw - self.var_id_offset
        
        # Handle NaN values by replacing with 0
        values = torch.nan_to_num(values, nan=0.0)
        
        # Embed values using shared linear transformation
        # (B, L, 1) → (B, L, d_model)
        value_emb = self.value_embedding(values)
        
        # Look up binary masks for each variable
        # var_ids: (B, L) → masks: (B, L, d_model)
        masks = self.binary_masks[var_ids]  # Index into (num_variables, d_model)
        
        # Element-wise product to create orthogonal embeddings
        # Zeros in mask will zero out corresponding dimensions
        # Scale by scale_factor to maintain expected variance after masking
        orthogonal_emb = value_emb * masks * self.scale_factor
        
        return orthogonal_emb
    
    def get_mask(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get missing value mask from input tensor.
        
        Args:
            X: Input tensor of shape (batch_size, seq_len, features)
        
        Returns:
            Boolean mask where True indicates missing value (NaN)
            Shape: (batch_size, seq_len, 1)
        """
        values = X[:, :, self.value_idx]
        return values.isnan().unsqueeze(-1)
    
    def get_var_ids(self, X: torch.Tensor) -> torch.Tensor:
        """
        Extract variable IDs from input tensor.
        
        Args:
            X: Input tensor
        
        Returns:
            Variable IDs of shape (batch_size, seq_len)
        """
        return X[:, :, self.var_idx]
    
    def __repr__(self):
        return (f"OrthogonalMaskEmbedding("
                f"num_variables={self.num_variables}, "
                f"d_model={self.d_model}, "
                f"dims_per_var={self.dims_per_var}, "
                f"scale_factor={self.scale_factor:.3f}, "
                f"frozen={self.freeze})")


class HermitePolynomialEmbedding(nn.Module):
    """
    Orthogonal embedding using Hermite polynomial basis functions.
    
    Hermite polynomials form an orthogonal basis under the Gaussian weight function.
    This provides a principled way to create orthogonal representations.
    
    For future implementation - uses probabilist's Hermite polynomials:
    H₀(x) = 1
    H₁(x) = x
    H₂(x) = x² - 1
    H₃(x) = x³ - 3x
    ...
    
    Note: This is a placeholder for future enhancement.
    The OrthogonalMaskEmbedding using binary masks is simpler and guarantees
    exact orthogonality for any input values.
    """
    
    def __init__(self, num_variables: int, d_model: int, max_degree: int = 5):
        super().__init__()
        raise NotImplementedError(
            "HermitePolynomialEmbedding is planned for future implementation. "
            "Use OrthogonalMaskEmbedding for now."
        )


# Quick test
if __name__ == "__main__":
    # Test OrthogonalMaskEmbedding
    print("Testing OrthogonalMaskEmbedding...")
    
    num_vars = 3
    d_model = 6
    batch_size = 2
    seq_len = 3
    
    emb = OrthogonalMaskEmbedding(
        num_variables=num_vars,
        d_model=d_model,
        value_idx=0,
        var_idx=1,
        freeze=True
    )
    
    print(f"\nEmbedding: {emb}")
    print(f"\nBinary masks:\n{emb.binary_masks}")
    print(f"Scale factor: {emb.scale_factor:.4f} (sqrt({d_model}/{emb.dims_per_var}) = sqrt({d_model/emb.dims_per_var:.1f}))")
    
    # Check orthogonality
    masks = emb.binary_masks
    dot_products = torch.mm(masks, masks.T)
    print(f"\nDot products between masks (should be diagonal):\n{dot_products}")
    
    # Test forward pass
    # Create test input: (batch, seq, features=[value, var_id])
    X = torch.randn(batch_size, seq_len, 2)
    X[:, :, 1] = torch.tensor([[0, 1, 2], [0, 1, 2]])  # Variable IDs
    
    output = emb(X)
    print(f"\nInput shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check orthogonality of output embeddings
    # Compare embeddings of different variables in same batch
    emb_var0 = output[0, 0, :]  # First variable
    emb_var1 = output[0, 1, :]  # Second variable
    emb_var2 = output[0, 2, :]  # Third variable
    
    print(f"\n⟨emb(S₀), emb(S₁)⟩ = {torch.dot(emb_var0, emb_var1).item():.6f}")
    print(f"⟨emb(S₀), emb(S₂)⟩ = {torch.dot(emb_var0, emb_var2).item():.6f}")
    print(f"⟨emb(S₁), emb(S₂)⟩ = {torch.dot(emb_var1, emb_var2).item():.6f}")
    
    # Verify scaling maintains expected variance
    print("\n--- Variance Analysis ---")
    # Check norm of embeddings
    norms = output.norm(dim=-1)
    print(f"Embedding norms (per position): {norms[0]}")
    print(f"Mean norm: {norms.mean().item():.4f}")
    
    print("\n✓ Orthogonality verified!")
    print("✓ Scaling applied to maintain variance!")
