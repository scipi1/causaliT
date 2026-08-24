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

Architecture (n_layers >= 2):
    decoder_output (d_model)
        → MLP Block: Linear(d_model, d_hidden) → Act → Dropout → Linear(d_hidden, d_model) → Dropout
        → Residual: + decoder_output
        → Final Projection: Linear(d_model, out_dim)

    The MLP block maps d_model → d_hidden → d_model with a residual connection,
    following the same pattern as transformer FFN sublayers. The final projection
    maps d_model → out_dim without residual.

    Standard expansion ratio is 2× (d_hidden = 2 * d_model), matching efficient
    transformer designs. No extra LayerNorm is added because the decoder already
    applies use_final_norm before this head.

Architecture (n_layers=1, backward compatible):
    d_model → Linear → out_dim  (equivalent to nn.Linear, no residual)

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
    Multi-layer perceptron output head with residual connection.
    
    Separates the MLP block (d_model → d_hidden → d_model, with residual)
    from the final projection (d_model → out_dim, no residual).
    
    Architecture (n_layers=2, default for MLP):
        x_res = x
        x = Linear(d_model, d_hidden) → activation → dropout
        x = Linear(d_hidden, d_model) → dropout
        x = x + x_res                    # residual connection
        x = Linear(d_model, out_dim)     # final projection
    
    Architecture (n_layers=3):
        x_res = x
        x = Linear(d_model, d_hidden) → activation → dropout
        x = Linear(d_hidden, d_hidden) → activation → dropout
        x = Linear(d_hidden, d_model) → dropout
        x = x + x_res                    # residual connection
        x = Linear(d_model, out_dim)     # final projection
    
    Architecture (n_layers=1, backward compatible):
        x = Linear(d_model, out_dim)     # single projection, no residual
    
    Args:
        d_model: Input dimension (transformer hidden dimension)
        out_dim: Output dimension per token (typically 1 for scalar predictions)
        n_layers: Number of linear layers in the MLP block.
                  1 = linear only (backward compatible, no MLP block).
                  2 = one hidden layer with activation + residual.
                  3+ = deeper MLP block + residual.
        d_hidden: Hidden dimension for MLP block. Defaults to 2 * d_model if None.
                  Standard transformer practice: 2-4× d_model.
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
        self.d_hidden = d_hidden if d_hidden is not None else (2 * d_model)
        
        # Select activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'gelu'.")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        
        if n_layers == 1:
            # Backward compatible: single linear projection, no MLP block
            self.mlp_block = None
            self.projection = nn.Linear(d_model, out_dim, bias=bias)
        else:
            # MLP block: d_model → d_hidden → ... → d_model (with residual)
            block_layers = []
            # First layer: d_model → d_hidden
            block_layers.append(nn.Linear(d_model, self.d_hidden, bias=bias))
            # Middle layers: d_hidden → d_hidden (for n_layers >= 3)
            for _ in range(n_layers - 2):
                block_layers.append(nn.Linear(self.d_hidden, self.d_hidden, bias=bias))
            # Final block layer: d_hidden → d_model (back to residual dimension)
            block_layers.append(nn.Linear(self.d_hidden, d_model, bias=bias))
            self.mlp_block = nn.ModuleList(block_layers)
            
            # Final projection: d_model → out_dim (no residual)
            self.projection = nn.Linear(d_model, out_dim, bias=bias)
    
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
            return self.projection(x)
        
        # MLP block with residual connection
        residual = x
        
        # Apply hidden layers with activation + dropout
        for layer in self.mlp_block[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final block layer: d_hidden → d_model (no activation, just dropout)
        x = self.mlp_block[-1](x)
        x = self.dropout(x)
        
        # Residual connection (d_model + d_model)
        x = x + residual
        
        # Final projection to output dimension (no residual)
        x = self.projection(x)
        return x
    
    def __repr__(self):
        return (
            f"MLPHead(d_model={self.d_model}, out_dim={self.out_dim}, "
            f"n_layers={self.n_layers}, d_hidden={self.d_hidden}, "
            f"residual={self.n_layers >= 2})"
        )


class PerNodeMLPHead(nn.Module):
    """
    Per-node MLP output head (DAGMA-style): one independent MLP per variable.

    Each variable i has its own decoder MLP:
        Linear(d_model -> d_hidden) -> activation -> Linear(d_hidden -> out_dim)

    The forward pass receives the transformer output (B, L, d_model) and the
    variable-ID column (B, L) and routes each token to its variable's MLP.
    This makes the decoding function node-specific, so the only cross-node
    mixing in the entire architecture is the structural attention itself.

    Args:
        d_model: Input dimension (transformer hidden dimension).
        out_dim: Output dimension per token (typically 1 for scalar predictions).
        num_variables: Number of variables (nodes) in the dataset.
        d_hidden: Hidden width of each per-node MLP.  Default 32.
        activation: Activation function ("relu", "gelu").  Default "relu".
        dropout: Dropout rate between hidden layers.  Default 0.0.
        bias: Whether to use bias in linear layers.  Default True.
        var_id_offset: Variable IDs are 1-indexed in SCM datasets (0 = padding),
            so the ID is shifted by this offset before indexing the MLP list.
            Default 1.
    """

    ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        num_variables: int,
        d_hidden: int = 32,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
        var_id_offset: int = 1,
    ):
        super().__init__()
        assert activation in self.ACTIVATIONS, (
            f"Invalid activation '{activation}'. Choose from {list(self.ACTIVATIONS)}."
        )
        self.d_model = d_model
        self.out_dim = out_dim
        self.num_variables = num_variables
        self.d_hidden = d_hidden
        self.var_id_offset = var_id_offset

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # One independent decoder MLP per variable (node).
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden, bias=bias),
                self.ACTIVATIONS[activation](),
                self.dropout,
                nn.Linear(d_hidden, out_dim, bias=bias),
            )
            for _ in range(num_variables)
        ])

    def forward(self, x: torch.Tensor, var_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) transformer output.
            var_ids: (B, L) variable IDs (1-indexed; 0 = padding).

        Returns:
            (B, L, out_dim) per-node predictions.
        """
        B, L, _ = x.shape
        # Shift to 0-indexed for ModuleList lookup; clamp padding to 0.
        idx = (var_ids.long() - self.var_id_offset).clamp(min=0, max=self.num_variables - 1)

        # Run every MLP on the full (B, L) batch and select the correct output
        # per token.  Vectorised; avoids a Python loop over the batch.
        all_outs = torch.stack(
            [mlp(x) for mlp in self.mlps], dim=2
        )  # (B, L, num_variables, out_dim)

        # Gather the output of the MLP matching each token's variable ID.
        out = torch.gather(
            all_outs,
            dim=2,
            index=idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.out_dim),
        ).squeeze(2)  # (B, L, out_dim)

        return out

    def __repr__(self):
        return (
            f"PerNodeMLPHead(d_model={self.d_model}, out_dim={self.out_dim}, "
            f"num_variables={self.num_variables}, d_hidden={self.d_hidden})"
        )
