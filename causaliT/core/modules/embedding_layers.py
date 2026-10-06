import torch
from torch import nn


class SinusoidalPosition(nn.Module):
    """
    Sinusoidal positional embedding 
    used in "Attention is all you need" (https://arxiv.org/abs/1706.03762)
    Embedding type: absolute & fixed
    """
    
    
    def __init__(self, max_pos:int, embed_dim:int, device):
        super().__init__()
        
        n = 10000.0 # internal variable, this n=10000 in the original paper
        
        assert embed_dim % 2 == 0, AssertionError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(embed_dim))
        
        positions = torch.arange(0, max_pos).unsqueeze_(1)
        denominators = torch.pow(n, 2*torch.arange(0, embed_dim//2)/embed_dim) # 10000^(2i/d_model), i is the index of embedding
        
        self.embeddings = torch.zeros(max_pos, embed_dim, device=device)
        self.embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        self.embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))
        
    def forward(self, x: torch.Tensor):
        return self.embeddings[x]



class identity_emb(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, X: torch.Tensor):
        indentity_fun = nn.Identity(device=self.device, dtype=torch.float32)
        return indentity_fun(X.unsqueeze(-1))




class nn_embedding(nn.Module):
    def __init__(self, num_embeddings,embedding_dim, device, *args, **kwargs):
        super().__init__()
        self.embed_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, device=device, dtype=torch.float32, *args, **kwargs)
    
    def forward(self, X: torch.Tensor):
        if self.embed_dim == 0:
            return torch.empty((X.shape[0], X.shape[1],0), device=X.get_device())
        else:
            X = X.to(torch.long)
            return self.embedding(X)



class linear_emb(nn.Module):
    def __init__(self, input_dim, embedding_dim, device):
        super().__init__()
        self.embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim, device=device, dtype=torch.float32)
        
    def forward(self, X: torch.Tensor):
        return self.embedding(X.unsqueeze(-1))
        


class mlp_emb(nn.Module):
    """
    Small shared MLP value embedding: nonlinear alternative to linear_emb.
    One hidden layer: Linear(input_dim -> hidden_dim) -> activation -> Linear(hidden_dim -> embedding_dim).
    The same weights are applied to every node (shared across nodes), exactly like linear_emb.
    """
    ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}

    def __init__(self, input_dim, embedding_dim, device, hidden_dim=64, activation="gelu"):
        super().__init__()
        assert activation in self.ACTIVATIONS, f"Invalid activation '{activation}'. Choose from {list(self.ACTIVATIONS)}."
        self.embedding = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, device=device, dtype=torch.float32),
            self.ACTIVATIONS[activation](),
            nn.Linear(in_features=hidden_dim, out_features=embedding_dim, device=device, dtype=torch.float32),
        )

    def forward(self, X: torch.Tensor):
        return self.embedding(X.unsqueeze(-1))
        


class linear_per_node_emb(nn.Module):
    """
    Per-node LINEAR value embedding: one independent linear map per variable.

    Each variable j has its own map:
        out = x_j * w_j + b_j        w_j, b_j in R^embedding_dim

    i.e. the scalar value is broadcast onto a per-node direction.  Unlike
    ``mlp_per_node_emb`` there is NO hidden nonlinearity, so the value stream
    keeps the raw-value contrast: two variables cannot be made perceptually
    similar by a learned encoder, and the attention weight A_ij couples
    directly to x_j (NOTEARS/DAGMA-style first-layer behaviour, up to the
    per-node direction).

    The forward pass receives the scalar value column (B, L) and the
    variable-ID column (B, L) and routes each token to its variable's map
    (same interface as ``mlp_per_node_emb``).

    Args:
        input_dim: Dimension of the scalar value input (typically 1).
        embedding_dim: Output dimension (must equal d_model for the value stream).
        num_variables: Number of variables (nodes) in the dataset.
        device: Torch device.
        var_id_offset: Variable IDs are 1-indexed in SCM datasets (0 = padding),
            so the ID is shifted by this offset before indexing.  Default 1.
        learnable: If True (default) the per-node directions/scales are
            learnable; if False they are frozen at the random init (pure
            fixed random expansion of the raw value).
        bias: If True, learn a per-node bias vector.  Default False (values
            are assumed centred; a bias would only add a constant to the
            attention output).
        dropout: Dropout rate applied to the embedded value output (train
            mode only).  Default 0.0 (disabled; nn.Identity, no behaviour
            change).
    """

    def __init__(
        self,
        input_dim,
        embedding_dim,
        num_variables,
        device,
        var_id_offset=1,
        learnable=True,
        bias=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_variables = num_variables
        self.var_id_offset = var_id_offset

        # Per-node direction init: random unit vectors (scale ~ |x_j| at init).
        w = torch.randn(num_variables, input_dim, embedding_dim,
                        device=device, dtype=torch.float32)
        w = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self.weight = nn.Parameter(w, requires_grad=learnable)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(num_variables, embedding_dim,
                            device=device, dtype=torch.float32),
                requires_grad=learnable,
            )
        else:
            self.bias = None

        # Dropout on the embedded value output (stateless; Identity keeps
        # dropout=0.0 a no-op).
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, values: torch.Tensor, var_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: (B, L) scalar values.
            var_ids: (B, L) variable IDs (1-indexed; 0 = padding).

        Returns:
            (B, L, embedding_dim) per-node linearly embedded values.
        """
        # Shift to 0-indexed for lookup; clamp padding to 0.
        idx = (var_ids.long() - self.var_id_offset).clamp(min=0, max=self.num_variables - 1)

        w = self.weight[idx]                       # (B, L, input_dim, embedding_dim)
        out = (values.unsqueeze(-1).unsqueeze(-1) * w).sum(dim=2)  # (B, L, embedding_dim)
        if self.bias is not None:
            out = out + self.bias[idx]
        return self.dropout(out)


class mlp_per_node_emb(nn.Module):
    """
    Per-node MLP value embedding (DAGMA-style): one independent MLP per variable.

    Each variable j has its own MLP:
        Linear(input_dim -> hidden_dim) -> activation -> Dropout -> Linear(hidden_dim -> embedding_dim)

    The forward pass receives the scalar value column (B, L) and the variable-ID
    column (B, L) and routes each token to its variable's MLP.  This lets every
    node learn its own nonlinear value functional while the hidden_dim stays
    fixed (does not scale with the number of nodes).

    Args:
        input_dim: Dimension of the scalar value input (typically 1).
        embedding_dim: Output dimension (must equal d_model for the value stream).
        num_variables: Number of variables (nodes) in the dataset.
        device: Torch device.
        hidden_dim: Hidden width of each per-node MLP (fixed, independent of
            the number of nodes).  Default 32.
        activation: Activation function ("gelu", "relu", "tanh").  Default "gelu".
        var_id_offset: Variable IDs are 1-indexed in SCM datasets (0 = padding),
            so the ID is shifted by this offset before indexing the MLP list.
            Default 1.
        dropout: Dropout rate applied after the hidden activation (train mode
            only).  Default 0.0 (disabled; nn.Identity, no behaviour change).
    """
    ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}

    def __init__(
        self,
        input_dim,
        embedding_dim,
        num_variables,
        device,
        hidden_dim=32,
        activation="gelu",
        var_id_offset=1,
        dropout=0.0,
    ):
        super().__init__()
        assert activation in self.ACTIVATIONS, (
            f"Invalid activation '{activation}'. Choose from {list(self.ACTIVATIONS)}."
        )
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        self.var_id_offset = var_id_offset

        # Dropout after the hidden activation (stateless, so one module is
        # shared by all per-node MLPs; Identity keeps dropout=0.0 a no-op).
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # One independent MLP per variable (node).
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim, device=device, dtype=torch.float32),
                self.ACTIVATIONS[activation](),
                self.dropout,
                nn.Linear(hidden_dim, embedding_dim, device=device, dtype=torch.float32),
            )
            for _ in range(num_variables)
        ])

    def forward(self, values: torch.Tensor, var_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: (B, L) scalar values.
            var_ids: (B, L) variable IDs (1-indexed; 0 = padding).

        Returns:
            (B, L, embedding_dim) per-node embedded values.
        """
        B, L = values.shape
        # Shift to 0-indexed for ModuleList lookup; clamp padding to 0.
        idx = (var_ids.long() - self.var_id_offset).clamp(min=0, max=self.num_variables - 1)

        # Run every MLP on the full (B, L) batch and select the correct output
        # per token.  This is vectorised and avoids a Python loop over the batch.
        # values: (B, L) -> (B, L, 1) -> (B, L, hidden) -> (B, L, emb)
        all_outs = torch.stack(
            [mlp(values.unsqueeze(-1)) for mlp in self.mlps], dim=2
        )  # (B, L, num_variables, embedding_dim)

        # Gather the output of the MLP matching each token's variable ID.
        out = torch.gather(
            all_outs,
            dim=2,
            index=idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.embedding_dim),
        ).squeeze(2)  # (B, L, embedding_dim)

        return out
