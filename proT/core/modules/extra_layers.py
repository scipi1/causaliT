import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np







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