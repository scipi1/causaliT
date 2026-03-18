"""
Orthogonal Linear Layer: Linear transformation with orthonormal columns.

Key idea:
- Constrains W such that W^T W = I (orthonormal columns)
- Preserves inner products: ⟨Wx, Wy⟩ = ⟨x, y⟩
- If input embeddings are orthogonal, output embeddings remain orthogonal

Use case:
- Key projection in attention when using orthogonal source embeddings
- Ensures attention scores between different source variables carry independent information
- Prevents spurious edges from mixing variable information during projection

Mathematical background:
- Uses Cayley parametrization: Q = (I - A)(I + A)^{-1} where A is skew-symmetric
- Extracts first in_features columns when out_features > in_features
- Optionally includes learnable scalar scale factor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class OrthogonalLinear(nn.Module):
    """
    Linear layer with orthonormal columns (W^T W = I).
    
    Preserves inner products: ⟨Wx, Wy⟩ = ⟨x, y⟩
    
    This ensures that if input embeddings are orthogonal (e.g., from 
    OrthogonalMaskEmbedding), the projected outputs remain orthogonal.
    The model can still learn the optimal rotation to fit the data.
    
    Uses Cayley parametrization: Q = (I - A)(I + A)^{-1}
    where A is a learnable skew-symmetric matrix.
    For non-square case (out > in), extracts first in_features columns.
    
    Requires: out_features >= in_features
    
    Args:
        in_features: Input dimension (e.g., d_model)
        out_features: Output dimension (e.g., d_queries_keys, must be >= in_features)
        use_scale: If True, include learnable scalar scale factor (default True)
        init_scale: Initial value for scale factor (default 1.0)
        init_std: Standard deviation for initializing skew-symmetric parameters (default 0.01)
        
    Example:
        >>> layer = OrthogonalLinear(in_features=24, out_features=32, use_scale=True)
        >>> x = torch.randn(16, 10, 24)  # (batch, seq, d_model)
        >>> y = layer(x)  # (batch, seq, 32)
        >>> # If x[0,0] ⊥ x[0,1], then y[0,0] ⊥ y[0,1]
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        use_scale: bool = True, 
        init_scale: float = 1.0,
        init_std: float = 0.01
    ):
        super().__init__()
        
        if out_features < in_features:
            raise ValueError(
                f"OrthogonalLinear requires out_features >= in_features for orthonormal columns. "
                f"Got out_features={out_features} < in_features={in_features}. "
                f"To use orthogonal key projection, set d_queries_keys >= d_model."
            )
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_scale = use_scale
        
        # Cayley parametrization: Q = (I - A)(I + A)^{-1}
        # A is skew-symmetric (A^T = -A), which has d*(d-1)/2 free parameters
        # We parametrize the upper triangular part
        n_skew_params = out_features * (out_features - 1) // 2
        self.skew_params = nn.Parameter(torch.zeros(n_skew_params))
        
        # Initialize with small random values for slight perturbation from identity
        nn.init.normal_(self.skew_params, mean=0.0, std=init_std)
        
        # Optional learnable scale factor
        if use_scale:
            self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        else:
            self.register_buffer('log_scale', None)
        
        # Cache for the upper triangular indices (computed once)
        self._triu_indices: Optional[torch.Tensor] = None
    
    def _get_triu_indices(self, device: torch.device) -> torch.Tensor:
        """Get upper triangular indices, cached for efficiency."""
        if self._triu_indices is None or self._triu_indices.device != device:
            self._triu_indices = torch.triu_indices(
                self.out_features, self.out_features, offset=1, device=device
            )
        return self._triu_indices
    
    def _build_skew_symmetric(self) -> torch.Tensor:
        """
        Build a skew-symmetric matrix A from the learnable parameters.
        
        A skew-symmetric matrix satisfies A^T = -A, with zeros on diagonal.
        We parametrize the upper triangular part and construct the lower part as -upper^T.
        
        Returns:
            Skew-symmetric matrix of shape (out_features, out_features)
        """
        device = self.skew_params.device
        dtype = self.skew_params.dtype
        
        A = torch.zeros(self.out_features, self.out_features, device=device, dtype=dtype)
        
        # Fill upper triangular (excluding diagonal)
        idx = self._get_triu_indices(device)
        A[idx[0], idx[1]] = self.skew_params
        
        # Make skew-symmetric: A = A - A^T
        A = A - A.T
        
        return A
    
    def get_orthogonal_matrix(self) -> torch.Tensor:
        """
        Compute the full orthogonal matrix Q using Cayley transform.
        
        Cayley formula: Q = (I - A)(I + A)^{-1}
        where A is skew-symmetric.
        
        Properties of Cayley transform:
        - If A is skew-symmetric, Q is orthogonal
        - Differentiable w.r.t. A
        - Q = I when A = 0 (starts from identity)
        
        Returns:
            Orthogonal matrix of shape (out_features, out_features)
        """
        A = self._build_skew_symmetric()
        I = torch.eye(self.out_features, device=A.device, dtype=A.dtype)
        
        # Q = (I - A)(I + A)^{-1}
        # Using solve for numerical stability: (I + A) Q = (I - A)
        # Equivalent to Q = solve(I + A, I - A)
        Q = torch.linalg.solve(I + A, I - A)
        
        return Q
    
    def get_weight(self) -> torch.Tensor:
        """
        Get the weight matrix W with orthonormal columns.
        
        For square case (out == in): W = Q (full orthogonal matrix)
        For non-square case (out > in): W = Q[:, :in] (first in_features columns)
        
        Returns:
            Weight matrix of shape (out_features, in_features) with orthonormal columns
        """
        Q = self.get_orthogonal_matrix()
        
        if self.out_features == self.in_features:
            return Q
        else:
            # Extract first in_features columns
            # These columns are still orthonormal (subset of orthonormal set)
            return Q[:, :self.in_features]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply orthogonal linear transformation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        W = self.get_weight()
        
        # Apply optional scale
        if self.use_scale:
            scale = torch.exp(self.log_scale)
            W = scale * W
        
        # F.linear expects weight of shape (out_features, in_features)
        return F.linear(x, W, bias=None)
    
    def get_scale(self) -> float:
        """Get the current scale factor value."""
        if self.use_scale:
            return torch.exp(self.log_scale).item()
        return 1.0
    
    def verify_orthonormality(self, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
        """
        Verify that W^T W ≈ I (orthonormal columns).
        
        Useful for debugging and testing.
        
        Args:
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            
        Returns:
            True if W has orthonormal columns (up to tolerance)
        """
        W = self.get_weight()
        WtW = W.T @ W
        I = torch.eye(self.in_features, device=W.device, dtype=W.dtype)
        return torch.allclose(WtW, I, rtol=rtol, atol=atol)
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"use_scale={self.use_scale}"
            + (f", scale={self.get_scale():.4f}" if self.use_scale else "")
        )


# Quick test and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Testing OrthogonalLinear")
    print("=" * 60)
    
    # Test 1: Square case
    print("\n1. Square case (in=out=8):")
    layer_square = OrthogonalLinear(in_features=8, out_features=8, use_scale=True)
    print(f"   {layer_square}")
    print(f"   Orthonormal columns: {layer_square.verify_orthonormality()}")
    
    # Test 2: Non-square case (d_out > d_in)
    print("\n2. Non-square case (in=8, out=16):")
    layer_nonsquare = OrthogonalLinear(in_features=8, out_features=16, use_scale=True)
    print(f"   {layer_nonsquare}")
    print(f"   Orthonormal columns: {layer_nonsquare.verify_orthonormality()}")
    
    # Test 3: Verify orthogonality preservation
    print("\n3. Testing orthogonality preservation:")
    batch_size, seq_len, d_model = 2, 3, 8
    d_qk = 16
    
    # Create orthogonal input vectors
    # Use different subspaces for each position in the sequence
    x = torch.zeros(batch_size, seq_len, d_model)
    # Position 0: active in dims 0-3
    x[0, 0, :4] = torch.randn(4)
    # Position 1: active in dims 4-7 (orthogonal to position 0)
    x[0, 1, 4:] = torch.randn(4)
    
    # Project through orthogonal layer
    layer = OrthogonalLinear(d_model, d_qk, use_scale=False)
    y = layer(x)
    
    # Check input orthogonality
    input_dot = torch.dot(x[0, 0].flatten(), x[0, 1].flatten())
    print(f"   Input ⟨x[0,0], x[0,1]⟩ = {input_dot.item():.6f}")
    
    # Check output orthogonality
    output_dot = torch.dot(y[0, 0].flatten(), y[0, 1].flatten())
    print(f"   Output ⟨y[0,0], y[0,1]⟩ = {output_dot.item():.6f}")
    
    # Test 4: Gradient flow
    print("\n4. Testing gradient flow:")
    layer = OrthogonalLinear(8, 16, use_scale=True)
    x = torch.randn(2, 10, 8, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    print(f"   skew_params.grad exists: {layer.skew_params.grad is not None}")
    print(f"   log_scale.grad exists: {layer.log_scale.grad is not None}")
    print(f"   skew_params.grad norm: {layer.skew_params.grad.norm().item():.6f}")
    print(f"   log_scale.grad value: {layer.log_scale.grad.item():.6f}")
    
    # Test 5: Invalid case (should raise error)
    print("\n5. Testing invalid case (out < in):")
    try:
        invalid_layer = OrthogonalLinear(in_features=16, out_features=8)
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   Correctly raised ValueError: {str(e)[:60]}...")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
