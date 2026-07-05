"""
Orthogonal Mask Embedding: Creates orthogonal embeddings using binary masks.

Key idea:
- Each variable gets a non-overlapping subset of dimensions (binary mask)
- Final embedding = value_embedding * binary_mask
- This ensures <emb(Si), emb(Sj)> = 0 for i != j

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

FixedOrthonormalEmbedding (see below) is an alternative that keeps exact
cross-variable orthogonality while using **dense** rows that span the full
d_model space (no idle dimensions), at the cost of being value-independent.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class OrthogonalMaskEmbedding(nn.Module):
    """
    Orthogonal embedding using binary masks and element-wise product.
    
    For variable Sj with value vj:
        emb(Sj) = value_embed(vj) * mask_j
    
    Where masks are orthogonal binary vectors that partition the d_model dimensions.
    
    Example for 3 variables with d_model=6:
        S1 mask: [1, 1, 0, 0, 0, 0]
        S2 mask: [0, 0, 1, 1, 0, 0]
        S3 mask: [0, 0, 0, 0, 1, 1]
    
    Args:
        num_variables: Number of source variables (e.g., 3 for S1, S2, S3)
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
                        ``(d_model - mask_start_dim) // num_variables`` -- this is
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
        
        # Shared value embedding: Linear(value_input_dim -> d_model)
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
        (+-1 for the first ``extra_dims`` variables) starting at ``mask_start_dim``.
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
        # We need 0-indexed for mask lookup (S1->0, S2->1, S3->2)
        var_ids = var_ids_raw - self.var_id_offset
        
        # Handle NaN values by replacing with 0
        values = torch.nan_to_num(values, nan=0.0)
        
        # Embed values using shared linear transformation
        # (B, L, 1) -> (B, L, d_model)
        value_emb = self.value_embedding(values)
        
        # Look up binary masks for each variable
        # var_ids: (B, L) -> masks: (B, L, d_model)
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


class FixedOrthonormalEmbedding(nn.Module):
    """
    Fixed, dense, mutually-orthonormal per-variable embedding ("orthogonal_fixed").

    Motivation
    ----------
    ``OrthogonalMaskEmbedding`` guarantees cross-variable orthogonality by giving
    each variable a *disjoint block* of ``dims_per_var = d_model // num_variables``
    dimensions.  With ``d_model`` not divisible by the variable count (or when two
    groups -- S and X -- must occupy equal-width disjoint partitions), some
    dimensions are left idle and every variable is confined to a low-dimensional
    axis-aligned subspace.

    This module instead assigns each variable a **dense row that spans all
    ``d_model`` dimensions**, with the rows constructed to be **mutually
    orthonormal** (``frame @ frame^T = I``, up to the optional ``scale``).  The
    frame is a **frozen buffer** (no trainable parameters), so orthogonality is
    preserved throughout training for free -- analogous to fixed sinusoidal
    positional encodings, but using a genuinely orthonormal frame.

    Identity-only (value-independent)
    ---------------------------------
    Unlike ``OrthogonalMaskEmbedding`` (whose key magnitude is value-modulated),
    this embedding depends ONLY on the variable identity -- the value column is
    ignored.  It is intended for the structural Q/K stream in SVFA mode, where the
    actual value still reaches the attention output through the separate value
    (V) stream.  The resulting attention scores are therefore constant across
    samples for a fixed set of projections (a clean, sample-independent adjacency).

    Joint S-perp-X frames
    ---------------------
    When S and X keys are concatenated (``sx_keys = cat([s_struct, xk_struct])``)
    ALL ``L_S + L_X`` rows must be mutually orthonormal, including S-vs-X.  To
    achieve this, two instances share the SAME underlying frame: construct both
    with the same ``total_variables``, ``d_model``, ``frame_type`` and ``seed``,
    and give them different ``row_offset`` values (0 for S, ``L_S`` for X).  Each
    instance deterministically regenerates the full ``(total_variables, d_model)``
    frame and keeps only its own row slice, so the two slices are guaranteed
    mutually orthogonal.  Requires ``total_variables <= d_model``.

    Args:
        num_variables: Number of variables this instance embeds (its own slice).
        d_model: Embedding dimension (rows are dense across all d_model dims).
        total_variables: Total number of variables across all instances that share
            the frame (defaults to ``num_variables`` for the single-group case).
            Must be ``<= d_model``.
        row_offset: Index of this instance's first row within the shared frame.
        value_idx: Index of the value feature (ignored; kept for API symmetry).
        var_idx: Index of the variable-ID feature in the input tensor.
        var_id_offset: Offset subtracted from 1-indexed var IDs before row lookup.
        frame_type: How the orthonormal frame is generated:
            ``"random"`` -- QR of a Gaussian matrix seeded by ``seed`` (default),
            ``"dct"``    -- DCT-II basis rows (deterministic, seed-independent).
        seed: Seed controlling the ``"random"`` frame.  In ``AttentionSelectorLayer``
            this is derived from the global training seed so the frame varies with
            the run seed while both S and X instances stay consistent.
        scale: Scalar multiplied into every row.  Preserves mutual orthogonality
            (only rescales norms); default 1.0 (unit-norm rows).
        device: Target device (buffer follows the parent module's ``.to(device)``).
    """

    def __init__(
        self,
        num_variables: int,
        d_model: int,
        total_variables: Optional[int] = None,
        row_offset: int = 0,
        value_idx: int = 0,
        var_idx: int = 1,
        var_id_offset: int = 1,
        frame_type: str = "random",
        seed: int = 0,
        scale: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()

        if total_variables is None:
            total_variables = num_variables

        if num_variables <= 0:
            raise ValueError(f"num_variables must be positive, got {num_variables}.")
        if total_variables > d_model:
            raise ValueError(
                f"FixedOrthonormalEmbedding needs d_model >= total_variables to build "
                f"mutually orthonormal rows, got total_variables={total_variables} > "
                f"d_model={d_model}."
            )
        if row_offset < 0 or row_offset + num_variables > total_variables:
            raise ValueError(
                f"row_offset={row_offset} + num_variables={num_variables} = "
                f"{row_offset + num_variables} exceeds total_variables={total_variables}."
            )
        if frame_type not in ("random", "dct"):
            raise ValueError(
                f"frame_type='{frame_type}' is invalid. Must be 'random' or 'dct'."
            )
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}.")

        self.num_variables = num_variables
        self.d_model = d_model
        self.total_variables = total_variables
        self.row_offset = row_offset
        self.value_idx = value_idx
        self.var_idx = var_idx
        self.var_id_offset = var_id_offset
        self.frame_type = frame_type
        self.seed = seed
        self.scale = scale
        self.device = device

        # Build the full shared frame, then slice this instance's rows.
        full_frame = self._build_frame(total_variables, d_model, frame_type, seed)
        rows = full_frame[row_offset:row_offset + num_variables].contiguous() * scale

        # Registered as a buffer -> frozen (never trained), moves with .to(device).
        self.frame: torch.Tensor
        self.register_buffer("frame", rows)

    @staticmethod
    def _build_frame(n: int, d: int, frame_type: str, seed: int) -> torch.Tensor:
        """Return an ``(n, d)`` tensor with mutually orthonormal rows.

        The full ``(n, d)`` frame is deterministic given ``(n, d, frame_type,
        seed)`` so that instances sharing these values (but differing in
        ``row_offset``) obtain consistent, mutually orthogonal row slices.
        """
        if frame_type == "random":
            # QR of a Gaussian matrix -> orthogonal columns.  Take the first n
            # columns and transpose to obtain n orthonormal rows of length d.
            g = torch.Generator().manual_seed(int(seed))
            a = torch.randn(d, d, generator=g)
            q, r = torch.linalg.qr(a)
            # Sign convention (make QR deterministic across BLAS backends):
            # flip columns so the diagonal of R is non-negative.
            signs = torch.sign(torch.diagonal(r))
            signs[signs == 0] = 1.0
            q = q * signs.unsqueeze(0)
            return q[:, :n].t().contiguous()          # (n, d), orthonormal rows

        # frame_type == "dct": DCT-II basis rows (deterministic, seed-independent).
        k = torch.arange(d, dtype=torch.float32)
        idx = torch.arange(n, dtype=torch.float32).unsqueeze(1)       # (n, 1)
        rows = torch.cos(math.pi * (k.unsqueeze(0) + 0.5) * idx / d)   # (n, d)
        rows = rows / rows.norm(dim=1, keepdim=True)                   # unit rows
        return rows.contiguous()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Look up the fixed orthonormal row for each token's variable ID.

        Args:
            X: Input tensor of shape (batch_size, seq_len, features).  Only the
               variable-ID column (``var_idx``) is used; the value column is
               ignored (identity-only embedding).

        Returns:
            Orthonormal embeddings of shape (batch_size, seq_len, d_model).
        """
        var_ids_raw = torch.nan_to_num(X[:, :, self.var_idx]).long()
        var_ids = var_ids_raw - self.var_id_offset
        # Clamp to the valid row range (datasets use 1-indexed IDs; 0 is padding).
        var_ids = var_ids.clamp(min=0, max=self.num_variables - 1)
        return self.frame[var_ids]

    def get_var_ids(self, X: torch.Tensor) -> torch.Tensor:
        """Extract variable IDs from the input tensor."""
        return X[:, :, self.var_idx]

    def __repr__(self):
        return (f"FixedOrthonormalEmbedding("
                f"num_variables={self.num_variables}, "
                f"d_model={self.d_model}, "
                f"total_variables={self.total_variables}, "
                f"row_offset={self.row_offset}, "
                f"frame_type={self.frame_type}, "
                f"scale={self.scale:.3f}, "
                f"seed={self.seed})")


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

    print("\n" + "=" * 60)
    print("Testing FixedOrthonormalEmbedding (dense, fixed, orthonormal rows)...")
    total_vars = 6
    d = 16
    emb_S = FixedOrthonormalEmbedding(num_variables=3, d_model=d, total_variables=total_vars,
                                      row_offset=0, frame_type="random", seed=123)
    emb_X = FixedOrthonormalEmbedding(num_variables=3, d_model=d, total_variables=total_vars,
                                      row_offset=3, frame_type="random", seed=123)
    full = torch.cat([emb_S.frame, emb_X.frame], dim=0)   # (6, 16)
    gram = full @ full.T
    print(f"\nFrame shape (S): {emb_S.frame.shape}, (X): {emb_X.frame.shape}")
    print(f"Gram of concatenated S+X rows (should be identity):\n{gram.round(decimals=4)}")
    off = gram - torch.diag(torch.diag(gram))
    print(f"Max off-diagonal |gram|: {off.abs().max().item():.3e}")
    print("\nDone.")
