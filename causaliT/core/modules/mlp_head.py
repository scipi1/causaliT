"""
MLP Output Head for causal transformers.

Replaces the single linear projection (de-embedding) with an optional multi-layer
perceptron to increase expressiveness for non-linear causal effect composition.

Motivation:
-----------
Structured attention (Toeplitz, CausalCross) blocks shortcut paths by design,
forcing indirect causal effects (e.g., S3 → X2 → X5) through multi-hop routing.
With a single decoder layer and linear output head, there is insufficient capacity
to compose these indirect non-linear effects.

The MLP head adds non-linearity at the output without compromising the causal
structure learned by attention. The attention still determines WHICH information
flows WHERE; the MLP determines HOW that information is transformed into predictions.

Design Choices:
- Hidden dim = d_ff (matches transformer convention)
- Default 2 layers: d_model → d_ff → out_dim
- ReLU activation (sparsity helps HSIC signal for causal discovery)
- No residual connection (this is a projection head, not a sublayer)
- n_layers=1 reduces to nn.Linear for backward compatibility

Staged Training Integration:
- Freeze MLP during causal initialization (structure learning phase)
- Unfreeze during main training to capture remaining non-linear variance
- Works with existing freeze_forecaster() / freeze_output_head() methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """
    Multi-layer perceptron output head.
    
    Replaces the single Linear forecaster/de-embedding layer with an MLP
    to increase expressiveness for non-linear causal effect composition.
    
    Architecture (n_layers=2, default):
        d_model → Linear → activation → dropout → Linear → out_dim
    
    Architecture (n_layers=3):
        d_model → Linear → activation → dropout → Linear → activation → dropout → Linear → out_dim
    
    Architecture (n_layers=1, backward compatible):
        d_model → Linear → out_dim  (equivalent to nn.Linear)
    
    Args:
        d_model: Input dimension (transformer hidden dimension)
        out_dim: Output dimension per token (typically 1 for scalar predictions)
        n_layers: Number of linear layers. 1 = linear only (backward compatible).
                  2 = one hidden layer with activation. 3+ = deeper MLP.
        d_hidden: Hidden dimension. Defaults to d_model if None.
        activation: Activation function ('relu', 'gelu'). Default 'relu'.
        dropout: Dropout rate between hidden layers. Default 0.0.
        bias: Whether to use bias in linear layers. Default True.
              Note: the original forecaster used bias=False. With n_layers=1,
              this parameter controls whether the single linear layer has bias.
    """
    
    def __init__(
        self,
        d_model: int,
        out_dim: int,
        n_layers: int = 2,
        d_hidden: int = None,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.d_hidden = d_hidden if d_hidden is not None else d_model
        
        # Select activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'gelu'.")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Build layers
        if n_layers == 1:
            # Backward compatible: single linear projection
            self.layers = nn.ModuleList([
                nn.Linear(d_model, out_dim, bias=bias)
            ])
        elif n_layers >= 2:
            layers = []
            # First layer: d_model → d_hidden
            layers.append(nn.Linear(d_model, self.d_hidden, bias=bias))
            # Middle layers: d_hidden → d_hidden
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(self.d_hidden, self.d_hidden, bias=bias))
            # Final layer: d_hidden → out_dim
            layers.append(nn.Linear(self.d_hidden, out_dim, bias=bias))
            self.layers = nn.ModuleList(layers)
        else:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP head.
        
        Args:
            x: (B, L, d_model) transformer output
            
        Returns:
            (B, L, out_dim) predictions
        """
        if self.n_layers == 1:
            # Single linear projection (backward compatible)
            return self.layers[0](x)
        
        # Multi-layer: apply activation + dropout between hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final layer: no activation (raw output)
        x = self.layers[-1](x)
        return x
    
    def __repr__(self):
        return (
            f"MLPHead(d_model={self.d_model}, out_dim={self.out_dim}, "
            f"n_layers={self.n_layers}, d_hidden={self.d_hidden})"
        )
