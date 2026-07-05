import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


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


class BatchConsistentKeyDropout(nn.Module):
    """
    Batch-consistent key (column) dropout for attention matrices.

    Unlike ``nn.Dropout``, which drops individual elements independently per
    sample, this module zeroes entire key positions (columns of the attention
    matrix) using a **single binary mask drawn once per forward call**.
    The same subset of keys is dropped for every sample in the batch, so that
    all B samples experience the same simplified parent set — a requirement for
    meaningful HSIC estimation over the batch.

    The drop probability ``p`` can be linearly annealed over training steps::

        p(t) = p_init + (p_final - p_init) * min(t / annealing_batches, 1)

    where ``t`` counts the number of training-mode forward calls (one per batch
    step).  The counter is saved in a registered buffer so checkpointing
    preserves annealing progress.

    No 1/(1-p) rescaling is applied: the goal is structural sparsification, not
    expectation-preserving noise injection.

    ``blanking_value`` controls how dropped keys are zeroed:

    * ``0.0`` (default) — multiply the attention matrix by the key mask after
      the activation (ReLU-Tanh, Sigmoid, Toeplitz).  Dropped positions become
      exactly 0.
    * ``float('-inf')`` — additive fill applied to **pre-softmax scores** before
      ``torch.softmax``.  The caller must pass the *score* tensor, not the
      post-softmax ``att``.  After softmax, dropped positions are exactly 0 and
      the remaining probability mass renormalises automatically.

    Attributes exposed after each training forward (``None`` in eval mode,
    ``p == 0``, or before the first forward):

    ``_last_key_mask``       : ``(L_S,)`` bool — True for kept keys.
    ``_last_active_queries`` : ``(L_X,)`` bool — True for queries that received
                               at least one non-zero key weight.

    Args:
        p_init            : Initial drop probability (0 → disabled).
        p_final           : Final drop probability after annealing.
                            ``None`` → constant ``p_init`` (no annealing).
        annealing_batches : Number of training-forward calls over which to
                            anneal from ``p_init`` to ``p_final``.
                            ``None`` or ``0`` → no annealing.
        blanking_value    : Value written into dropped key positions.
                            ``0.0`` for post-activation matrices;
                            ``float('-inf')`` for pre-softmax score tensors.
    """

    def __init__(
        self,
        p_init: float,
        p_final: Optional[float] = None,
        annealing_batches: Optional[int] = None,
        blanking_value: float = 0.0,
    ):
        super().__init__()

        if not (0.0 <= p_init <= 1.0):
            raise ValueError(f"p_init must be in [0, 1], got {p_init}")
        if p_final is not None and not (0.0 <= p_final <= 1.0):
            raise ValueError(f"p_final must be in [0, 1], got {p_final}")

        self.p_init = float(p_init)
        self.p_final = float(p_final) if p_final is not None else None
        self.annealing_batches = (
            int(annealing_batches)
            if annealing_batches is not None and int(annealing_batches) > 0
            else None
        )
        self.blanking_value = float(blanking_value)
        self._use_annealing = (
            self.p_final is not None and self.annealing_batches is not None
        )

        # Current effective drop probability — updated each training step.
        self.p = self.p_init

        # Persistent step counter — only registered as a buffer (and therefore
        # persisted in state_dict / checkpoints) when annealing is actually
        # active.  When annealing is off, ``_step_count`` is a plain attribute
        # so that checkpoints produced by BKD-enabled stages do not inject a
        # ``batch_key_dropout._step_count`` key that would cause a
        # strict state_dict mismatch when loading into a stage that has
        # ``batch_key_dropout=None`` (i.e. no BKD module at all).
        if self._use_annealing:
            self.register_buffer("_step_count", torch.tensor(0, dtype=torch.long))
        else:
            self._step_count = torch.tensor(0, dtype=torch.long)

        # Outputs of the most recent training forward (None in eval mode / p=0).
        self._last_key_mask: Optional[torch.Tensor] = None
        self._last_active_queries: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _current_p(self) -> float:
        """Return the linearly annealed drop probability at the current step."""
        if not self._use_annealing:
            return self.p_init
        progress = min(1.0, self._step_count.item() / self.annealing_batches)
        return self.p_init + progress * (self.p_final - self.p_init)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply batch-consistent key dropout.

        Args:
            x : Attention weight or score tensor.
                Shape ``(B, L_X, L_S)`` or ``(B, H, L_X, L_S)``.
                For ``blanking_value=0.0``  pass the post-activation ``att``.
                For ``blanking_value=-inf`` pass the pre-softmax ``scores``.

        Returns:
            Tensor of the same shape with dropped key columns blanked.
            Returned unchanged (no copy) when in eval mode or ``p == 0``.
        """
        if not self.training:
            self._last_key_mask = None
            self._last_active_queries = None
            return x

        # Advance step counter and recompute effective p before sampling.
        self.p = self._current_p()
        self._step_count += 1

        if self.p <= 0.0:
            self._last_key_mask = None
            self._last_active_queries = None
            return x

        S_dim = x.shape[-1]
        # One Boolean mask for the whole batch: True = keep, False = drop.
        keep_mask = (torch.rand(S_dim, device=x.device) >= self.p)  # (L_S,)

        if self.blanking_value == 0.0:
            x = x * keep_mask.to(x.dtype)
        else:
            # Additive masking — used for pre-softmax scores with -inf blanking.
            x = x.masked_fill(~keep_mask, self.blanking_value)

        # Derive per-query activity (post-masking): which Xi had ≥1 active key?
        # For -inf blanking the caller applies softmax afterwards, so the
        # row-sum check is done on the raw (potentially -inf-filled) scores.
        # Using finite-only sum: treat -inf entries as 0 for the activity check.
        x_det = x.detach()
        if self.blanking_value != 0.0:
            x_det = torch.nan_to_num(x_det, neginf=0.0)

        row_sums = x_det.sum(dim=-1)          # (B, [H,] L_X)
        if row_sums.dim() == 3:               # multihead: average over H
            row_sums = row_sums.mean(dim=1)   # (B, L_X)
        # Batch-consistent → first sample is representative for all.
        self._last_active_queries = row_sums[0] > 0   # (L_X,) bool
        self._last_key_mask = keep_mask                # (L_S,) bool

        return x

    # ------------------------------------------------------------------
    def get_hsic_active_mask(self) -> Optional[torch.Tensor]:
        """
        Return the ``(L_X,)`` bool mask of query variables that received at
        least one active key in the most recent training forward pass.

        Returns ``None`` when the feature is disabled (eval mode, ``p == 0``,
        or no training forward has been called yet), signalling the forecaster
        to include all variables in the HSIC computation unchanged.

        Typical usage in the forecaster::

            bcd = layer.global_cross_attention.inner_attention \\
                       .batch_consistent_dropout
            active_x = bcd.get_hsic_active_mask() if bcd is not None else None
            if active_x is not None and not active_x.all():
                residuals_for_hsic = residuals_per_x[:, active_x]
        """
        return self._last_active_queries
