import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dag_decisiveness_loss(
    phi: torch.Tensor, 
    tau: torch.Tensor = None,
    exclude_diagonal: bool = True,
    eps: float = 1e-6  # Increased from 1e-8 to prevent 0 * -inf = NaN when p ≈ 1
) -> torch.Tensor:
    """
    Compute decisiveness loss for DAG edge probabilities.
    
    This loss encourages the network to make decisive edge decisions by penalizing
    edge probabilities near 0.5 (maximum uncertainty). The loss is based on the
    binary entropy of edge probabilities:
    
        H(p) = -p*log(p) - (1-p)*log(1-p)
    
    Binary entropy is maximized at p=0.5 and minimized at p=0 or p=1.
    By minimizing this entropy, we push edge probabilities away from 0.5,
    encouraging the network to commit to edge presence (p→1) or absence (p→0).
    
    Args:
        phi: DAG logits tensor, shape (L, S) for single-head or (H, L, S) for multi-head.
             These are raw logits before sigmoid.
        tau: Optional Gumbel-Softmax temperature. If provided, probabilities are
             computed as sigmoid(phi/tau) to match the training-time distribution.
             If None, uses sigmoid(phi) directly.
        exclude_diagonal: If True, excludes diagonal elements from the loss computation.
                         This is important for self-attention DAGs where diagonal
                         represents self-loops which should be zero anyway.
        eps: Small constant for numerical stability in log computation.
    
    Returns:
        Scalar tensor: Mean binary entropy of edge probabilities.
                      Lower values indicate more decisive edges (closer to 0 or 1).
    
    Example usage in loss function:
        # Get phi from attention layer
        phi = attention_layer.inner_attention.phi
        tau = torch.exp(attention_layer.inner_attention.log_tau)
        
        # Compute decisiveness loss
        decisiveness = dag_decisiveness_loss(phi, tau, exclude_diagonal=True)
        
        # Add to total loss with weight
        total_loss = reconstruction_loss + lambda_decisive * decisiveness
    
    Notes:
        - For DAGMaskAntisym, the diagonal of phi is already 0 by construction,
          but excluding it prevents counting sigmoid(0)=0.5 in the entropy.
        - The temperature tau affects the sharpness of probabilities. Higher tau
          leads to probabilities closer to 0.5, which this loss will penalize.
        - This loss implicitly encourages temperature annealing as the network
          learns, since lower tau leads to lower entropy for the same phi values.
    """
    # Compute edge probabilities
    if tau is not None:
        # Use temperature-scaled sigmoid to match Gumbel-Softmax distribution
        tau_clamped = tau.clamp(min=0.01)
        p = torch.sigmoid(phi / tau_clamped)
    else:
        p = torch.sigmoid(phi)
    
    # Create diagonal mask if needed
    if exclude_diagonal and phi.dim() >= 2:
        if phi.dim() == 2:
            # Single-head: (L, S)
            L, S = phi.shape
            if L == S:  # Only for square matrices
                diag_mask = ~torch.eye(L, S, device=phi.device, dtype=torch.bool)
                p = p[diag_mask]
        elif phi.dim() == 3:
            # Multi-head: (H, L, S)
            H, L, S = phi.shape
            if L == S:  # Only for square matrices
                diag_mask = ~torch.eye(L, S, device=phi.device, dtype=torch.bool)
                diag_mask = diag_mask.unsqueeze(0).expand(H, -1, -1)
                p = p[diag_mask]
    
    # Compute binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    # Use clamping to avoid log(0)
    p_clamped = p.clamp(min=eps, max=1-eps)
    entropy = -p_clamped * torch.log(p_clamped) - (1 - p_clamped) * torch.log(1 - p_clamped)
    
    return entropy.mean()


def dag_temperature_loss(log_tau: torch.Tensor, target_tau: float = 0.1) -> torch.Tensor:
    """
    Compute temperature penalty to encourage annealing.
    
    This loss encourages the Gumbel-Softmax temperature to decrease over training,
    leading to sharper (more discrete) edge masks. The loss is simply the
    current temperature value, so minimizing it pushes tau toward zero.
    
    However, we use a soft lower bound to prevent tau from becoming too small
    (which could cause numerical issues or overly hard masks too early).
    
    Args:
        log_tau: Log of the Gumbel-Softmax temperature (learned parameter).
        target_tau: Target temperature value. The loss is the ReLU of (tau - target_tau),
                   so no penalty is applied when tau <= target_tau.
    
    Returns:
        Scalar tensor: Temperature penalty.
    
    Example usage in loss function:
        log_tau = attention_layer.inner_attention.log_tau
        tau_loss = dag_temperature_loss(log_tau, target_tau=0.1)
        total_loss = reconstruction_loss + lambda_tau * tau_loss
    
    Notes:
        - Using log_tau as the parameter ensures tau > 0.
        - The target_tau provides a soft floor: once tau reaches target_tau,
          no further penalty is applied.
        - Typical values: target_tau=0.1 for sharp masks, target_tau=0.5 for softer.
    """
    tau = torch.exp(log_tau)
    # Only penalize if tau > target_tau
    return F.relu(tau - target_tau)


def dag_sparsity_loss(
    phi: torch.Tensor,
    tau: torch.Tensor = None,
    exclude_diagonal: bool = True,
    lambda_decisive: float = 1.0,
    lambda_tau: float = 0.1,
    target_tau: float = 0.1,
    log_tau: torch.Tensor = None
) -> tuple:
    """
    Combined DAG sparsity loss: decisiveness + temperature annealing.
    
    This function combines the decisiveness loss (entropy minimization) and
    temperature penalty into a single convenience function.
    
    Args:
        phi: DAG logits tensor from attention layer.
        tau: Current Gumbel-Softmax temperature value.
        exclude_diagonal: Whether to exclude diagonal from decisiveness loss.
        lambda_decisive: Weight for decisiveness (entropy) loss.
        lambda_tau: Weight for temperature penalty.
        target_tau: Target temperature for annealing.
        log_tau: Log of temperature (required if lambda_tau > 0).
    
    Returns:
        Tuple of (total_loss, decisiveness_loss, tau_loss):
            - total_loss: Weighted sum of all components
            - decisiveness_loss: Binary entropy of edge probabilities
            - tau_loss: Temperature penalty
    
    Example usage:
        phi = attention.inner_attention.phi
        log_tau = attention.inner_attention.log_tau
        tau = torch.exp(log_tau)
        
        total, decisive, tau_penalty = dag_sparsity_loss(
            phi, tau, 
            exclude_diagonal=True,
            lambda_decisive=0.1,
            lambda_tau=0.01,
            log_tau=log_tau
        )
        loss = reconstruction_loss + total
    """
    # Compute decisiveness loss
    decisive_loss = dag_decisiveness_loss(phi, tau, exclude_diagonal)
    
    # Compute temperature penalty
    if lambda_tau > 0 and log_tau is not None:
        tau_loss = dag_temperature_loss(log_tau, target_tau)
    else:
        tau_loss = torch.tensor(0.0, device=phi.device)
    
    # Combined loss
    total = lambda_decisive * decisive_loss + lambda_tau * tau_loss
    
    return total, decisive_loss, tau_loss







class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps) * self.scale
        x = x / n * self.g
        return x


class Normalization(nn.Module):
    def __init__(self, method, d_model=None):
        super().__init__()
        assert method in ["layer", "scale", "batch", "power", "MBN", "MLN", "MBPN", "MLPN","none"]
        if method == "layer":
            assert d_model
            self.norm = nn.LayerNorm(d_model)
        elif method == "scale":
            self.norm = ScaleNorm(d_model)
        
        elif method == "MBN":
            self.norm = MaskedBatchNorm1d(d_model)
            
        elif method == "MLN":
            self.norm = MaskedLayerNorm(d_model)
            
        elif method == "MBPN":
            self.norm = MaskedBatchPowerNorm(d_model)
            
        elif method == "MLPN":
            self.norm = MaskedLayerPowerNorm(d_model)
        
        # not needed now
        # elif method == "power":
        #     self.norm = MaskPowerNorm(d_model, warmup_iters=1000)
        
        elif method == "none":
            self.norm = NoNorm
        else:
            assert d_model
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method
        
    def forward(self, x,*args, **kwargs):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        
        elif self.method == "layer":
            return self.norm(x)
        
        return self.norm(x, *args, **kwargs)
    
    
    
    
def NoNorm(x,*args, **kwargs):
    """
    No normalization
    """
    return x




class DAGMask(nn.Module):
    """
    DAG mask for LieAttention with learnable phi parameter.
    
    This is the original independent parameterization where each edge (i,j)
    is parameterized independently. This allows bidirectional edges and 
    doesn't enforce DAG structure constraints.
    
    Args:
        n_heads: Number of attention heads
        query_seq_len: Length of query sequence
        key_seq_len: Length of key sequence
        init_std: Standard deviation for random initialization of phi.
                  Default 0.1 gives initial probabilities roughly in [0.3, 0.7].
                  Use 0 for zero initialization (starts at P=0.5 everywhere).
    """
    def __init__(self, n_heads, query_seq_len, key_seq_len, init_std: float = 0.1):
        super(DAGMask, self).__init__()
        if n_heads > 1:
            self.phi = nn.Parameter(torch.randn(n_heads, query_seq_len, key_seq_len) * init_std)
        else:
            self.phi = nn.Parameter(torch.randn(query_seq_len, key_seq_len) * init_std)
    
    def forward(self, attention_scores: torch.Tensor, mask: torch.Tensor = None, mask_val=-float("inf")):
        """
        For compatibility with mask_layer interface.
        Currently just returns attention_scores as is.
        """
        return attention_scores


class DAGMaskAntisym(nn.Module):
    """
    Antisymmetric DAG mask with structural constraints.
    
    Instead of parameterizing phi_ij and phi_ji independently, uses a single
    learnable upper-triangular matrix W and derives antisymmetric logits:
    
        W_antisym = W_upper - W_upper.T
        phi = W_antisym  (logits)
        P(i→j) = sigmoid(W_antisym[i,j])
        
    Properties:
    - P(i→j) + P(j→i) = 1 (mutual exclusivity / competition between directions)
    - Diagonal is automatically 0 (no self-loops)
    - Parameters reduced from L² to L(L-1)/2
    
    This is more structurally correct for DAG learning as it prevents the model
    from assigning high probability to both i→j and j→i simultaneously.
    
    Args:
        n_heads: Number of attention heads
        query_seq_len: Length of query sequence (must equal key_seq_len)
        key_seq_len: Length of key sequence (must equal query_seq_len)
        init_std: Standard deviation for random initialization of W.
                  Default 0.1 gives initial edge probabilities roughly in [0.3, 0.7].
                  Use 0 for zero initialization (starts at P=0.5 everywhere).
    """
    def __init__(self, n_heads, query_seq_len, key_seq_len, init_std: float = 0.1):
        super(DAGMaskAntisym, self).__init__()
        
        # For now, only support square attention (self-attention DAGs)
        # Cross-attention DAGs are bipartite and don't need antisymmetry
        assert query_seq_len == key_seq_len, \
            "DAGMaskAntisym only supports square attention (query_seq_len == key_seq_len). " \
            "For cross-attention, use DAGMask instead."
        
        self.n_vars = query_seq_len
        self.n_heads = n_heads
        
        # Parameterize only upper triangular entries (i < j)
        # Number of parameters: n_vars * (n_vars - 1) / 2 per head
        # Initialize with small random values to break symmetry
        if n_heads > 1:
            # Shape: (H, n_vars, n_vars) but only upper triangular is used
            self.W = nn.Parameter(torch.randn(n_heads, self.n_vars, self.n_vars) * init_std)
        else:
            # Shape: (n_vars, n_vars) but only upper triangular is used
            self.W = nn.Parameter(torch.randn(self.n_vars, self.n_vars) * init_std)
        
        # Create a buffer for the upper triangular mask (not learnable)
        triu_mask = torch.triu(torch.ones(self.n_vars, self.n_vars), diagonal=1)
        self.register_buffer('triu_mask', triu_mask)
        
        # Expose phi as a property for compatibility with existing code
        # that reads self.phi directly
        
    @property
    def phi(self) -> torch.Tensor:
        """
        Returns the antisymmetric logits for DAG edge probabilities.
        
        The antisymmetric structure ensures:
        - phi[i,j] = -phi[j,i] for all i ≠ j
        - phi[i,i] = 0 (no self-loops)
        
        Returns:
            torch.Tensor: Antisymmetric logits, shape (L, S) or (H, L, S) for multi-head.
        """
        return self._get_antisym_logits()
    
    def _get_antisym_logits(self) -> torch.Tensor:
        """
        Compute antisymmetric logits from the learnable upper triangular W.
        
        W_upper = triu(W, diagonal=1)  # zero out diagonal and lower
        W_antisym = W_upper - W_upper.T  # antisymmetric: W[i,j] = -W[j,i]
        """
        if self.n_heads > 1:
            # For multi-head: W shape is (H, n_vars, n_vars)
            # Apply upper triangular mask per head
            W_upper = self.W * self.triu_mask.unsqueeze(0)  # (H, n_vars, n_vars)
            W_antisym = W_upper - W_upper.transpose(-2, -1)  # (H, n_vars, n_vars)
        else:
            # For single-head: W shape is (n_vars, n_vars)
            W_upper = self.W * self.triu_mask  # (n_vars, n_vars)
            W_antisym = W_upper - W_upper.T  # (n_vars, n_vars)
        
        return W_antisym
    
    def get_dag_probabilities(self) -> torch.Tensor:
        """
        Returns the posterior probability of each edge being active in the learned DAG.
        
        Due to antisymmetric parameterization:
        - P(i→j) = sigmoid(W_antisym[i,j])
        - P(j→i) = sigmoid(W_antisym[j,i]) = sigmoid(-W_antisym[i,j]) = 1 - P(i→j)
        - P(i→i) = 0.5 (from sigmoid(0)), but in practice diagonal should be masked
        
        Returns:
            torch.Tensor: Edge probabilities in [0, 1], shape (L, S) or (H, L, S).
        """
        return torch.sigmoid(self.phi)
    
    def get_dag_logits(self) -> torch.Tensor:
        """
        Returns the antisymmetric logits (phi) of the learned DAG.
        
        Returns:
            torch.Tensor: Antisymmetric logits, shape (L, S) or (H, L, S).
        """
        return self.phi
    
    def forward(self, attention_scores: torch.Tensor, mask: torch.Tensor = None, mask_val=-float("inf")):
        """
        For compatibility with mask_layer interface.
        Currently just returns attention_scores as is.
        """
        return attention_scores


class UniformAttentionMask(nn.Module):
    def __init__(self) -> None:
        super(UniformAttentionMask,self).__init__()
    
    def forward(self, attention_scores:torch.Tensor, mask:torch.Tensor,mask_val=-float("inf")):
        """
        Applies masking to the attention scores.
        
        Args:
        - attention_scores: Tensor of shape (batch_size, N_queries, N_keys).
        - mask: Boolean tensor of shape (N_keys), where False means the corresponding key should be masked (zeroed).
        
        Returns:
        - masked_attention_scores: Tensor with masked attention scores.
        """

        assert attention_scores.shape[-1] == len(mask), AssertionError(f"Got mask of length {len(mask)}, expected {attention_scores.shape[-1]}")
        
        # Ensure the mask is a torch tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        
        # Ensure the mask is on the same device as the attention scores
        if mask.device != attention_scores.device:
            mask = mask.to(attention_scores.device)
        
        # Convert boolean mask to float and expand it to match attention_scores
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N_keys)
        mask=mask.expand_as(attention_scores)
        # Apply the mask to zero out the attention scores where mask is False
        
        return attention_scores.masked_fill(mask, mask_val)
    
class NAIMAttentionMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, attention_scores:torch.Tensor, mask:torch.Tensor,mask_val=-torch.inf):
        """
        Applies masking to the attention scores.
        
        Args:
        - attention_scores: Tensor of shape (batch_size, N_queries, N_keys).
        - mask: Boolean tensor of shape (N_keys), where False means the corresponding key should be masked (zeroed).
        
        Returns:
        - masked_attention_scores: Tensor with masked attention scores.
        """

        assert attention_scores.shape[-1] == len(mask), AssertionError(f"Got mask of length {len(mask)}, expected {attention_scores.shape[-1]}")
        
        # Ensure the mask is a torch tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        
        # Ensure the mask is on the same device as the attention scores
        if mask.device != attention_scores.device:
            mask = mask.to(attention_scores.device)
        
        # Convert boolean mask to float and expand it to match attention_scores
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N_keys)
        mask=mask.expand_as(attention_scores)
        # Apply the mask to zero out the attention scores where mask is False
        
        return attention_scores.masked_fill(torch.isnan(attention_scores), mask_val)
    
    
    
    
class MaskedLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias   = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x, pad_mask):
        """
        x        : (B, L, D)
        pad_mask : (B, L)  bool, True = real token
        """
        # reshape mask for broadcasting
        m = pad_mask.float()            # (B, L, 1)

        # number of real tokens per position (0 or 1 here)
        denom = m.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, L, 1)

        # compute mean / var over hidden dims ONLY for real tokens
        mean = (x * m).sum(dim=-1, keepdim=True) / denom
        var  = ((x - mean)**2 * m).sum(dim=-1, keepdim=True) / denom

        x_hat = (x*m - mean) / torch.sqrt(var + self.eps)
        
        breakpoint()
        return self.weight * x_hat + self.bias
    
    
class MaskedLayerPowerNorm(nn.Module):
    def __init__(self, d_model, p_init=2.0, eps=1e-5):
        super().__init__()
        self.gamma  = nn.Parameter(torch.ones(d_model))
        self.beta   = nn.Parameter(torch.zeros(d_model))
        self.log_p  = nn.Parameter(torch.log(torch.tensor(p_init)))
        self.eps    = eps

    def forward(self, x, mask):
        """
        x    : (B, L, D)  – embedded input sequence
        mask : (B, L)     – True for real token
        """
        m = mask.float()              # (B, L, 1)

        # avoid div-by-0 if an entire sequence is padding
        denom = m.sum(dim=-1, keepdim=True).clamp(min=1.0)

        mu_token = (x * m).sum(dim=-1, keepdim=True) / denom

        p = torch.exp(self.log_p)
        dev_p   = ((x - mu_token).abs().pow(p) * m).sum(dim=-1, keepdim=True) / denom
        sigma_p     = dev_p.pow(1.0 / p)

        x_norm = (x - mu_token) / (sigma_p + self.eps)
        
        
        breakpoint()
        return self.gamma * x_norm + self.beta
    
    

    
    
    
class MaskedBatchNorm1d(nn.Module):
    """
    BatchNorm1d that excludes padding tokens from batch statistics.

    Args
    ----
    d_model : int   # hidden size (feature dimension)
    eps     : float
    momentum: float # same meaning as in nn.BatchNorm1d
    """
    def __init__(self, d_model, eps=1e-5, momentum=0.1):
        super().__init__()
        self.d_model  = d_model
        self.eps      = eps
        self.momentum = momentum

        # learnable scale & shift (γ, β)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias   = nn.Parameter(torch.zeros(d_model))

        # running stats for inference (BN semantics)
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var",  torch.ones(d_model))

    def forward(self, x, mask):
        """
        x    : (B, L, D)  embedded tokens
        mask : (B, L)     bool → True for *real* token, False for padding
        """
        B, L, D = x.shape
        x_flat  = x.view(-1, D)         # (B·L, D)
        m_flat  = mask.view(-1)         # (B·L,)

        # pick only the visible rows
        visible = x_flat[m_flat]        # (N_vis, D)  might be empty

        if self.training and visible.numel():
            mean = visible.mean(dim=0)              # (D,)
            var  = visible.var(dim=0, unbiased=False)

            # update running stats
            self.running_mean = \
                (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var  = \
                (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean = self.running_mean
            var  = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # broadcast
        return self.weight * x_norm + self.bias
    
    
    

class MaskedBatchPowerNorm(nn.Module):
    """
    Batch-style PowerNorm without centring.
    Statistics are computed on *visible* tokens only (mask == 1).
    """
    def __init__(self, d_model, p_init=2.0, eps=1e-5, momentum=0.1):
        super().__init__()
        
        self.gamma    = nn.Parameter(torch.ones(d_model))
        self.beta     = nn.Parameter(torch.zeros(d_model))
        self.log_p    = nn.Parameter(torch.log(torch.tensor(p_init)))
        self.eps      = eps
        self.momentum = momentum
        # running power statistic (for inference)
        self.register_buffer("running_pow", torch.ones(d_model))

    def forward(self, x, mask):
        """
        x    : (B, L, D)
        mask : (B, L)   True = real token
        """
        B, L, D = x.shape
        x_flat  = x.view(-1, D)               # (B·L, D)
        m_flat  = mask.view(-1)               # (B·L,)
        visible = x_flat[m_flat]              # rows that matter

        p    = torch.exp(self.log_p)

        if self.training and visible.numel():
            pow_batch = (visible.abs().pow(p).mean(dim=0) + self.eps).pow(1/p)
            if pow_batch.isnan().any():
                print("NaN in pow_batch")
                breakpoint()
            # EMA update
            self.running_pow = (1-self.momentum)*self.running_pow + self.momentum*pow_batch
            pow_stat = pow_batch
        else:
            pow_stat = self.running_pow

        x_norm = x / (pow_stat + self.eps)      # ← no centring
        return self.gamma * x_norm + self.beta