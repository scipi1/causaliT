from os.path import dirname, abspath
import sys
# ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
# sys.path.append(ROOT_DIR)
from math import sqrt, log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causaliT.core.modules.extra_layers import UniformAttentionMask
from causaliT.utils.entropy_utils import register_attention_entropy, calculate_attention_entropy
from typing import List



class LieAttention(nn.Module):
    """
    Lie Attention mechanism with DAG mask learning.
    
    Note: Acyclicity regularization (NOTEARS) only supports single-head attention.
          For multi-head attention, phi will have shape (H, L, S) and acyclicity 
          regularization should be disabled in the forecaster.
    
    The causal DAG is learned through a learnable tensor (phi) that models edge probabilities.
    A Gumbel-Softmax trick enables differentiable sampling from Bernoulli(sigmoid(phi)).
    Running averages of attention statistics are used as priors for KL regularization.
    """
    def __init__(self, mask_layer: nn.Module, attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(LieAttention, self).__init__()
        
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        
        # Store DAGMask as submodule to ensure proper parameter registration
        from causaliT.core.modules.extra_layers import DAGMask
        if isinstance(mask_layer, DAGMask):
            self.dag_mask = mask_layer  # Store as submodule for proper state_dict handling
            self.phi = self.dag_mask.phi  # Reference for convenience
            
            # Initialize running averages as buffers (not optimized, but saved in state_dict)
            # These track EMA statistics for monitoring and prior regularization
            self.register_buffer('runav_att_mean', torch.zeros_like(self.phi))
            self.register_buffer('runav_att_snr', torch.zeros_like(self.phi))
        else:
            self.dag_mask = None
            self.phi = None
            self.runav_att_mean = None
            self.runav_att_snr = None
        
        # Gumbel-Softmax temperature - learnable with annealing
        # Starts high (τ=2.0) for exploration, anneals toward low values for sharper masks
        self.log_tau = nn.Parameter(torch.tensor(log(2.0)))
        self.tau_min = 0.1  # Minimum temperature
        self.tau_max = 5.0  # Maximum temperature

        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
        
        # --- Lie commutator amplification (new) ---
        self.log_gain = nn.Parameter(torch.tensor(log(10.0)))   # start strong; set to 0.0 if you want gain=1
        self.log_tau_comm = nn.Parameter(torch.tensor(log(0.2)))  # tanh temperature (linear slope = gain/tau)
        self.max_gain = 1e3
        self.enforce_nonneg_flow = True  # set False if you want to allow negative flow
    
    def get_dag_probabilities(self) -> torch.Tensor:
        """
        Returns the posterior probability of each edge being active in the learned DAG.
        
        This is useful for:
        - Inference: Extract the learned causal structure
        - Visualization: Plot the learned DAG
        - Evaluation: Compare against ground-truth DAG
        
        Returns:
            torch.Tensor: Edge probabilities in [0, 1], shape (L, S) or (H, L, S) for multi-head.
                         Returns None if no DAG is being learned (phi is None).
        """
        if self.phi is not None:
            return torch.sigmoid(self.phi)
        return None
    
    def get_dag_logits(self) -> torch.Tensor:
        """
        Returns the raw logits (phi) of the learned DAG.
        
        Returns:
            torch.Tensor: Raw logits, shape (L, S) or (H, L, S) for multi-head.
                         Returns None if no DAG is being learned.
        """
        return self.phi
        
    def _update_running_average(self, att):
        """
        Update running averages of attention statistics.
        
        Args:
            att: Attention evidence tensor (batch_size, ...)
            
        Returns:
            tuple: (batch_mean, batch_snr) with gradients attached for regularization
        """
        alpha = 0.9
        
        # Compute batch statistics (keep gradients for regularization)
        batch_mean = torch.mean(att, dim=0)
        batch_std = torch.std(att, dim=0)
        batch_snr = batch_mean / (batch_std + 1e-6)
        
        # Update running averages (no gradients, in-place operations on buffers)
        if self.runav_att_mean is not None:
            with torch.no_grad():
                if (self.runav_att_mean != 0).any():
                    # EMA update using in-place operations
                    self.runav_att_mean.mul_(alpha).add_(batch_mean, alpha=1-alpha)
                    self.runav_att_snr.mul_(alpha).add_(batch_snr, alpha=1-alpha)
                else:
                    # First update: initialize with batch statistics
                    self.runav_att_mean.copy_(batch_mean)
                    self.runav_att_snr.copy_(batch_snr)
        
        # Return batch statistics with gradients for immediate use in regularization
        return batch_mean, batch_snr
        
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        ):
        """
        Forward pass for Lie Attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask_miss_k: Missing key mask
            mask_miss_q: Missing query mask
            pos: Positional encoding
            causal_mask: Whether to apply causal masking
            hard_mask: Optional hard mask tensor of shape (L, S) for single-head or (H, L, S) for multi-head.
                       Values should be in [0, 1], where 1 = attention allowed.
                       Applied as element-wise product with attention scores.
        """
        # Handle both single-head (3D) and multi-head (4D) tensors
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
            
        # convert to commutator
        comm = scores - scores.transpose(-1,-2)
        
        # gain-before-tanh amplifier
        gain = torch.exp(self.log_gain).clamp(1e-3, self.max_gain)
        tau_comm = torch.exp(self.log_tau_comm).clamp(1e-3, 10.0)

        # amplify small commutators (linear near 0, saturates to +/-1)
        comm_amp = torch.tanh((gain / tau_comm) * comm)

        # gate negatives toward 0 (your design choice)
        scores = F.gelu(comm_amp)
        
        # apply scaling
        att = scale * scores
        
        # Optionally enforce strict "negative -> no flow"
        if self.enforce_nonneg_flow:
            A = torch.relu(att)
        else:
            A = att

        A = torch.nan_to_num(self.dropout(A))
        
        # Update running average for monitoring and store batch statistics for regularization
        evidence = A
        batch_mean, batch_snr = self._update_running_average(evidence)
        
        # Store batch statistics as attributes for access by forecaster (with gradients)
        self.batch_att_mean = batch_mean
        self.batch_att_snr = batch_snr
        
        # Apply DAG mask if phi is available
        if self.phi is not None:
            # Get learnable temperature (clamped to safe range)
            tau = torch.exp(self.log_tau).clamp(self.tau_min, self.tau_max)
            
            # Sample batch DAG logits using Gumbel-Softmax trick
            u = torch.rand_like(self.phi)
            m_relaxed = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.phi) / tau)
            M = m_relaxed
            
            # Add batch dimension
            # For single-head: phi shape is (L, S) -> M becomes (1, L, S)
            # For multi-head: phi shape is (H, L, S) -> M becomes (1, H, L, S)
            M = M.unsqueeze(0)
            
            att = att * M
        
        # Apply hard mask if provided (ground-truth DAG structure)
        if hard_mask is not None:
            # hard_mask shape: (L, S) for single-head, (H, L, S) for multi-head
            # Expand to match attention shape: (B, L, S) or (B, H, L, S)
            if is_multihead:
                # hard_mask: (H, L, S) -> (1, H, L, S)
                if hard_mask.dim() == 2:
                    # Single mask for all heads: (L, S) -> (1, 1, L, S) -> broadcast
                    hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
                else:
                    # Per-head mask: (H, L, S) -> (1, H, L, S)
                    hard_mask = hard_mask.unsqueeze(0)
            else:
                # hard_mask: (L, S) -> (1, L, S)
                hard_mask = hard_mask.unsqueeze(0)
            
            att = att * hard_mask
        
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
            
            
        A = torch.nan_to_num(self.dropout(att))
        
        # Compute output values
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)
            
            
        return V.contiguous(), A, entropy



class CausalCrossAttention(nn.Module):
    """
    Causal Cross-Attention with DAG mask learning.
    
    This module implements cross-attention (query and key sequences may differ)
    adapted for causality, with learnable DAG structure.
    
    Features:
    - GeLU(Tanh) activation instead of Softmax for causality-friendly attention
    - Learnable DAG mask (phi) via Gumbel-Softmax trick
    - Running averages for KL prior regularization
    - Decoupled attention scores and causal structure
    
    The causal DAG is learned through a learnable tensor (phi) that models edge probabilities.
    A Gumbel-Softmax trick enables differentiable sampling from Bernoulli(sigmoid(phi)).
    Running averages of attention statistics are used as priors for KL regularization.
    
    Note: Cross-attention DAGs are bipartite (query → key), inherently acyclic,
          so NOTEARS regularization is not needed.
    """
    def __init__(self, mask_layer: nn.Module, attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(CausalCrossAttention, self).__init__()
        
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
        # Store DAGMask as submodule to ensure proper parameter registration
        from causaliT.core.modules.extra_layers import DAGMask
        if isinstance(mask_layer, DAGMask):
            self.dag_mask = mask_layer  # Store as submodule for proper state_dict handling
            self.phi = self.dag_mask.phi  # Reference for convenience
            
            # Initialize running averages as buffers (not optimized, but saved in state_dict)
            # These track EMA statistics for monitoring and prior regularization
            self.register_buffer('runav_att_mean', torch.zeros_like(self.phi))
            self.register_buffer('runav_att_snr', torch.zeros_like(self.phi))
        else:
            self.dag_mask = None
            self.phi = None
            self.runav_att_mean = None
            self.runav_att_snr = None
        
        # Gumbel-Softmax temperature - learnable with annealing
        # Starts high (τ=2.0) for exploration, anneals toward low values for sharper masks
        self.log_tau_gs = nn.Parameter(torch.tensor(log(2.0)))
        self.tau_gs_min = 0.1  # Minimum temperature
        self.tau_gs_max = 5.0  # Maximum temperature
        
        # Gain/temperature parameters for GeLU(Tanh) activation
        self.log_gain = nn.Parameter(torch.tensor(log(1.0)))
        self.log_tau = nn.Parameter(torch.tensor(log(0.2)))  # tanh temperature
        self.max_gain = 10.0
    
    def get_dag_probabilities(self) -> torch.Tensor:
        """
        Returns the posterior probability of each edge being active in the learned DAG.
        
        This is useful for:
        - Inference: Extract the learned causal structure
        - Visualization: Plot the learned DAG
        - Evaluation: Compare against ground-truth DAG
        
        Returns:
            torch.Tensor: Edge probabilities in [0, 1], shape (L, S) or (H, L, S) for multi-head.
                         Returns None if no DAG is being learned (phi is None).
        """
        if self.phi is not None:
            return torch.sigmoid(self.phi)
        return None
    
    def get_dag_logits(self) -> torch.Tensor:
        """
        Returns the raw logits (phi) of the learned DAG.
        
        Returns:
            torch.Tensor: Raw logits, shape (L, S) or (H, L, S) for multi-head.
                         Returns None if no DAG is being learned.
        """
        return self.phi
    
    def _update_running_average(self, att):
        """
        Update running averages of attention statistics.
        
        Args:
            att: Attention evidence tensor (batch_size, ...)
            
        Returns:
            tuple: (batch_mean, batch_snr) with gradients attached for regularization
        """
        alpha = 0.9
        
        # Compute batch statistics (keep gradients for regularization)
        batch_mean = torch.mean(att, dim=0)
        batch_std = torch.std(att, dim=0)
        batch_snr = batch_mean / (batch_std + 1e-6)
        
        # Update running averages (no gradients, in-place operations on buffers)
        if self.runav_att_mean is not None:
            with torch.no_grad():
                if (self.runav_att_mean != 0).any():
                    # EMA update using in-place operations
                    self.runav_att_mean.mul_(alpha).add_(batch_mean, alpha=1-alpha)
                    self.runav_att_snr.mul_(alpha).add_(batch_snr, alpha=1-alpha)
                else:
                    # First update: initialize with batch statistics
                    self.runav_att_mean.copy_(batch_mean)
                    self.runav_att_snr.copy_(batch_snr)
        
        # Return batch statistics with gradients for immediate use in regularization
        return batch_mean, batch_snr
    
    def _expand_hard_mask(self, hard_mask: torch.Tensor, is_multihead: bool) -> torch.Tensor:
        """Expand hard_mask to match scores shape and convert to additive mask."""
        if is_multihead:
            if hard_mask.dim() == 2:
                # Single mask for all heads: (L, S) -> (1, 1, L, S)
                hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
            else:
                # Per-head mask: (H, L, S) -> (1, H, L, S)
                hard_mask = hard_mask.unsqueeze(0)
        else:
            # hard_mask: (L, S) -> (1, L, S)
            hard_mask = hard_mask.unsqueeze(0)
        return hard_mask
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        ):
        """
        Forward pass for Causal Cross-Attention with DAG learning.
        
        Args:
            query: Query tensor (B, L, E) or (B, L, H, E) for multi-head
            key: Key tensor (B, S, E) or (B, S, H, E) for multi-head
            value: Value tensor (B, S, E) or (B, S, H, E) for multi-head
            mask_miss_k: Missing key mask (unused in simplified version)
            mask_miss_q: Missing query mask (unused in simplified version)
            pos: Positional encoding for causal masking
            causal_mask: Whether to apply causal masking
            hard_mask: Optional hard mask tensor of shape (L, S) or (H, L, S).
                       Values in [0, 1], where 1 = attention allowed.
                       Applied AFTER the learned DAG mask as element-wise product.
        """
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        scores = scale * scores
        
        # Apply causal mask (additive, before activation)
        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos, n_heads=H)
            scores = scores + M_causal
        
        gain = torch.exp(self.log_gain).clamp(1e-3, self.max_gain)
        tau = torch.exp(self.log_tau).clamp(1e-3, 10.0)
        
        # Causality-friendly activation: GeLU(Tanh(scores))
        att = F.gelu(F.tanh((gain / tau) * scores))
        
        # Handle NaN from all-masked rows
        att = torch.nan_to_num(att, nan=0.0)
        
        # Update running average for monitoring and store batch statistics for regularization
        evidence = att
        batch_mean, batch_snr = self._update_running_average(evidence)
        
        # Store batch statistics as attributes for access by forecaster (with gradients)
        self.batch_att_mean = batch_mean
        self.batch_att_snr = batch_snr
        
        # Apply learned DAG mask if phi is available
        if self.phi is not None:
            # Get learnable temperature (clamped to safe range)
            tau_gs = torch.exp(self.log_tau_gs).clamp(self.tau_gs_min, self.tau_gs_max)
            
            # Sample batch DAG logits using Gumbel-Softmax trick
            u = torch.rand_like(self.phi)
            m_relaxed = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.phi) / tau_gs)
            M = m_relaxed
            
            # Add batch dimension
            # For single-head: phi shape is (L, S) -> M becomes (1, L, S)
            # For multi-head: phi shape is (H, L, S) -> M becomes (1, H, L, S)
            M = M.unsqueeze(0)
            
            att = att * M
        
        # Apply hard mask if provided (ground-truth DAG structure)
        if hard_mask is not None:
            hard_mask_expanded = self._expand_hard_mask(hard_mask, is_multihead)
            att = att * hard_mask_expanded
        
        # Calculate entropy before dropout
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
        
        # Apply dropout
        A = self.dropout(att)
        
        # Compute output values
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)
        
        return V.contiguous(), A, entropy



class PhiSoftMax(nn.Module):
    """
    Softmax Attention with learnable DAG mask (phi).
    
    Combines standard softmax attention (from ScaledDotAttention) with 
    learnable DAG structure (from LieAttention/CausalCrossAttention).
    
    Key design: Both hard_mask and learned DAG mask (phi) are applied BEFORE 
    softmax as additive -inf masking to ensure proper information isolation.
    
    Features:
    - Standard softmax attention
    - Learnable DAG mask (phi) via Gumbel-Softmax trick
    - Running averages for KL prior regularization
    - Threshold-based conversion of soft phi to additive mask
    
    The causal DAG is learned through a learnable tensor (phi) that models edge probabilities.
    A Gumbel-Softmax trick enables differentiable sampling from Bernoulli(sigmoid(phi)).
    """
    def __init__(self, mask_layer: nn.Module, attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(PhiSoftMax, self).__init__()
        
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
        # Store DAGMask as submodule to ensure proper parameter registration
        from causaliT.core.modules.extra_layers import DAGMask
        if isinstance(mask_layer, DAGMask):
            self.dag_mask = mask_layer  # Store as submodule for proper state_dict handling
            self.phi = self.dag_mask.phi  # Reference for convenience
            
            # Initialize running averages as buffers (not optimized, but saved in state_dict)
            # These track EMA statistics for monitoring and prior regularization
            self.register_buffer('runav_att_mean', torch.zeros_like(self.phi))
            self.register_buffer('runav_att_snr', torch.zeros_like(self.phi))
        else:
            self.dag_mask = None
            self.phi = None
            self.runav_att_mean = None
            self.runav_att_snr = None
        
        # Gumbel-Softmax temperature - learnable with annealing
        # Starts high (τ=2.0) for exploration, anneals toward low values for sharper masks
        self.log_tau = nn.Parameter(torch.tensor(log(2.0)))
        self.tau_min = 0.1  # Minimum temperature
        self.tau_max = 5.0  # Maximum temperature
        
        # Threshold for converting soft phi to hard mask (values below threshold -> -inf)
        # This is learnable to allow the model to adjust the sparsity
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.threshold_min = 0.01
        self.threshold_max = 0.99
        
        # Large constant for soft thresholding (makes the transition sharp but differentiable)
        self.mask_scale = 1e4
    
    def get_dag_probabilities(self) -> torch.Tensor:
        """
        Returns the posterior probability of each edge being active in the learned DAG.
        
        This is useful for:
        - Inference: Extract the learned causal structure
        - Visualization: Plot the learned DAG
        - Evaluation: Compare against ground-truth DAG
        
        Returns:
            torch.Tensor: Edge probabilities in [0, 1], shape (L, S) or (H, L, S) for multi-head.
                         Returns None if no DAG is being learned (phi is None).
        """
        if self.phi is not None:
            return torch.sigmoid(self.phi)
        return None
    
    def get_dag_logits(self) -> torch.Tensor:
        """
        Returns the raw logits (phi) of the learned DAG.
        
        Returns:
            torch.Tensor: Raw logits, shape (L, S) or (H, L, S) for multi-head.
                         Returns None if no DAG is being learned.
        """
        return self.phi
    
    def _update_running_average(self, att):
        """
        Update running averages of attention statistics.
        
        Args:
            att: Attention evidence tensor (batch_size, ...)
            
        Returns:
            tuple: (batch_mean, batch_snr) with gradients attached for regularization
        """
        alpha = 0.9
        
        # Compute batch statistics (keep gradients for regularization)
        batch_mean = torch.mean(att, dim=0)
        batch_std = torch.std(att, dim=0)
        batch_snr = batch_mean / (batch_std + 1e-6)
        
        # Update running averages (no gradients, in-place operations on buffers)
        if self.runav_att_mean is not None:
            with torch.no_grad():
                if (self.runav_att_mean != 0).any():
                    # EMA update using in-place operations
                    self.runav_att_mean.mul_(alpha).add_(batch_mean, alpha=1-alpha)
                    self.runav_att_snr.mul_(alpha).add_(batch_snr, alpha=1-alpha)
                else:
                    # First update: initialize with batch statistics
                    self.runav_att_mean.copy_(batch_mean)
                    self.runav_att_snr.copy_(batch_snr)
        
        # Return batch statistics with gradients for immediate use in regularization
        return batch_mean, batch_snr
    
    def _expand_hard_mask(self, hard_mask: torch.Tensor, is_multihead: bool) -> torch.Tensor:
        """Expand hard_mask to match scores shape and convert to additive mask."""
        if is_multihead:
            if hard_mask.dim() == 2:
                # Single mask for all heads: (L, S) -> (1, 1, L, S)
                hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
            else:
                # Per-head mask: (H, L, S) -> (1, H, L, S)
                hard_mask = hard_mask.unsqueeze(0)
        else:
            # hard_mask: (L, S) -> (1, L, S)
            hard_mask = hard_mask.unsqueeze(0)
        return hard_mask
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        ):
        """
        Forward pass for PhiSoftMax Attention with learnable DAG.
        
        Args:
            query: Query tensor (B, L, E) or (B, L, H, E) for multi-head
            key: Key tensor (B, S, E) or (B, S, H, E) for multi-head
            value: Value tensor (B, S, E) or (B, S, H, E) for multi-head
            mask_miss_k: Missing key mask (unused in simplified version)
            mask_miss_q: Missing query mask (unused in simplified version)
            pos: Positional encoding for causal masking
            causal_mask: Whether to apply causal masking
            hard_mask: Optional hard mask tensor of shape (L, S) or (H, L, S).
                       Values in [0, 1], where 1 = attention allowed.
                       Applied BEFORE softmax via additive -inf masking.
        """
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        # Apply causal mask (additive, before softmax)
        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos, n_heads=H)
            scores = scores + M_causal
        
        # Apply learned DAG mask BEFORE softmax (if phi is available)
        if self.phi is not None:
            # Get learnable temperature (clamped to safe range)
            tau = torch.exp(self.log_tau).clamp(self.tau_min, self.tau_max)
            
            # Sample batch DAG logits using Gumbel-Softmax trick
            u = torch.rand_like(self.phi)
            gumbel_noise = torch.log(u + 1e-8) - torch.log(1 - u + 1e-8)
            m_relaxed = torch.sigmoid((gumbel_noise + self.phi) / tau)
            
            # Clamp threshold to valid range
            thresh = self.threshold.clamp(self.threshold_min, self.threshold_max)
            
            # Convert to additive mask using soft thresholding (differentiable)
            # Values below threshold get large negative values, above threshold get ~0
            # Using: -scale * relu(threshold - m_relaxed) 
            # This gives: 0 when m_relaxed >= threshold, negative when m_relaxed < threshold
            phi_mask_additive = -self.mask_scale * F.relu(thresh - m_relaxed)
            
            # Add batch dimension
            # For single-head: phi shape is (L, S) -> M becomes (1, L, S)
            # For multi-head: phi shape is (H, L, S) -> M becomes (1, H, L, S)
            phi_mask_additive = phi_mask_additive.unsqueeze(0)
            
            scores = scores + phi_mask_additive
        
        # Track all-masked rows for zeroing out after softmax
        all_masked_rows = None
        
        # Apply hard mask BEFORE softmax (additive, -inf for masked positions)
        # This ensures masked positions don't influence softmax normalization
        if hard_mask is not None:
            hard_mask_expanded = self._expand_hard_mask(hard_mask, is_multihead)
            
            # Detect all-masked rows (where entire row is 0 in the mask)
            # These rows would cause softmax([-inf, -inf, ...]) = NaN
            all_masked_rows = (hard_mask_expanded.sum(dim=-1, keepdim=True) == 0)
            
            # Convert 0/1 mask to additive mask: 0 -> -inf, 1 -> 0
            hard_mask_additive = torch.where(
                hard_mask_expanded == 0,
                torch.tensor(float('-inf'), device=scores.device, dtype=scores.dtype),
                torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            )
            
            # For all-masked rows, use zeros instead of -inf to avoid NaN in softmax
            # The attention output will be zeroed out afterwards
            if all_masked_rows.any():
                hard_mask_additive = torch.where(
                    all_masked_rows,
                    torch.tensor(0.0, device=scores.device, dtype=scores.dtype),
                    hard_mask_additive
                )
            
            scores = scores + hard_mask_additive
        
        # Scaled softmax (now safe - no all-inf rows)
        att = torch.softmax(scale * scores, dim=-1)
        
        # Zero out attention for all-masked rows (they shouldn't contribute)
        # This is done with multiplication to maintain gradient flow
        if all_masked_rows is not None and all_masked_rows.any():
            att = att * (~all_masked_rows).float()
        
        # Update running average for monitoring and store batch statistics for regularization
        if self.phi is not None:
            batch_mean, batch_snr = self._update_running_average(att)
            # Store batch statistics as attributes for access by forecaster (with gradients)
            self.batch_att_mean = batch_mean
            self.batch_att_snr = batch_snr
        
        # Calculate entropy before dropout
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
        
        # Apply dropout
        A = self.dropout(att)
        
        # Compute output values
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)
        
        return V.contiguous(), A, entropy


class ScaledDotAttention(nn.Module):
    """
    Simplified Scaled Dot-Product Attention.
    
    Hard mask is applied BEFORE softmax to ensure masked positions don't
    influence the softmax normalization (preventing information leakage).
    """
    def __init__(self, mask_layer: nn.Module, attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(ScaledDotAttention, self).__init__()
        
        self.mask_layer = mask_layer
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
    
    def _expand_hard_mask(self, hard_mask: torch.Tensor, is_multihead: bool) -> torch.Tensor:
        """Expand hard_mask to match scores shape and convert to additive mask."""
        if is_multihead:
            if hard_mask.dim() == 2:
                # Single mask for all heads: (L, S) -> (1, 1, L, S)
                hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
            else:
                # Per-head mask: (H, L, S) -> (1, H, L, S)
                hard_mask = hard_mask.unsqueeze(0)
        else:
            # hard_mask: (L, S) -> (1, L, S)
            hard_mask = hard_mask.unsqueeze(0)
        return hard_mask
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        ):
        """
        Forward pass for Scaled Dot-Product Attention.
        
        Args:
            query: Query tensor (B, L, E) or (B, L, H, E) for multi-head
            key: Key tensor (B, S, E) or (B, S, H, E) for multi-head
            value: Value tensor (B, S, E) or (B, S, H, E) for multi-head
            mask_miss_k: Missing key mask (unused in simplified version)
            mask_miss_q: Missing query mask (unused in simplified version)
            pos: Positional encoding for causal masking
            causal_mask: Whether to apply causal masking
            hard_mask: Optional hard mask tensor of shape (L, S) or (H, L, S).
                       Values in [0, 1], where 1 = attention allowed.
                       Applied BEFORE softmax via additive -inf masking.
        """
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        # Apply causal mask (additive, before softmax)
        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos, n_heads=H)
            scores = scores + M_causal
        
        # Track all-masked rows for zeroing out after softmax
        all_masked_rows = None
        
        # Apply hard mask BEFORE softmax (additive, -inf for masked positions)
        # This ensures masked positions don't influence softmax normalization
        if hard_mask is not None:
            hard_mask_expanded = self._expand_hard_mask(hard_mask, is_multihead)
            
            # Detect all-masked rows (where entire row is 0 in the mask)
            # These rows would cause softmax([-inf, -inf, ...]) = NaN
            all_masked_rows = (hard_mask_expanded.sum(dim=-1, keepdim=True) == 0)
            
            # Convert 0/1 mask to additive mask: 0 -> -inf, 1 -> 0
            hard_mask_additive = torch.where(
                hard_mask_expanded == 0,
                torch.tensor(float('-inf'), device=scores.device, dtype=scores.dtype),
                torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            )
            
            # For all-masked rows, use zeros instead of -inf to avoid NaN in softmax
            # The attention output will be zeroed out afterwards
            if all_masked_rows.any():
                hard_mask_additive = torch.where(
                    all_masked_rows,
                    torch.tensor(0.0, device=scores.device, dtype=scores.dtype),
                    hard_mask_additive
                )
            
            scores = scores + hard_mask_additive
        
        # Scaled softmax (now safe - no all-inf rows)
        att = torch.softmax(scale * scores, dim=-1)
        
        # Zero out attention for all-masked rows (they shouldn't contribute)
        # This is done with multiplication to maintain gradient flow
        if all_masked_rows is not None and all_masked_rows.any():
            att = att * (~all_masked_rows).float()
        
        # Calculate entropy before dropout
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
        
        # Apply dropout
        A = self.dropout(att)
        
        # Compute output values
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)
        
        return V.contiguous(), A, entropy


class ScaledDotAttentionNAIM(nn.Module):
    """
    Scaled Dot-Product Attention with NAIM (Not All Is Missing) handling.
    
    This version includes special handling for missing data via mask_miss_k and mask_miss_q,
    using ReLU after softmax to handle the missing query mask.
    Reference: https://arxiv.org/abs/2407.11540
    
    NOTE: The hard_mask in this version is applied AFTER softmax, which can cause
    information leakage. Use ScaledDotAttention for proper causal masking.
    """
    def __init__(self, mask_layer: nn.Module, attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(ScaledDotAttentionNAIM, self).__init__()
        
        self.mask_layer = mask_layer
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        ):
        """
        Forward pass for Scaled Dot-Product Attention with NAIM missing data handling.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask_miss_k: Missing key mask (True = missing)
            mask_miss_q: Missing query mask (True = missing)
            pos: Positional encoding
            causal_mask: Whether to apply causal masking
            hard_mask: Optional hard mask tensor of shape (L, S) for single-head or (H, L, S) for multi-head.
                       Values should be in [0, 1], where 1 = attention allowed.
                       WARNING: Applied AFTER softmax - may cause information leakage.
        """
        # Handle both single-head (3D) and multi-head (4D) tensors
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        # Apply causal mask
        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos, n_heads=H)
            scores = scores + M_causal
        
        # Apply missing data masks
        # masking missing value with -inf to force the softmax to zero
        # (reference https://arxiv.org/abs/2407.11540)
        
        if is_multihead:
            # For multi-head: scores shape is (B, H, L, S)
            key_size = scores.size(-1)  # S
            query_size = scores.size(-2)  # L
            
            # Expand masks to (B, H, L, S)
            if mask_miss_k is not None:
                mask_miss_k_expanded = mask_miss_k.unsqueeze(1).expand(-1, H, -1, -1).expand(-1, -1, -1, query_size).transpose(-1, -2)
            
            if mask_miss_q is not None:
                mask_miss_q_expanded = mask_miss_q.unsqueeze(1).expand(-1, H, -1, -1).expand(-1, -1, -1, key_size)
        else:
            # For single-head: scores shape is (B, L, S)
            key_size = scores.size(-1)  # S
            query_size = scores.size(-2)  # L
            
            if mask_miss_k is not None:
                mask_miss_k_expanded = mask_miss_k.expand(-1, -1, query_size).transpose(-1, -2)
            
            if mask_miss_q is not None:
                mask_miss_q_expanded = mask_miss_q.expand(-1, -1, key_size)
        
        if mask_miss_k is not None:
            M_k = torch.zeros_like(scores).masked_fill_(mask_miss_k_expanded, -torch.inf)
        else:
            M_k = torch.zeros_like(scores)
            
        if mask_miss_q is not None:
            M_q = torch.zeros_like(scores).masked_fill_(mask_miss_q_expanded, -torch.inf)
        else:
            M_q = torch.zeros_like(scores)
        
        att = torch.relu(torch.softmax(scale * (scores + M_k), dim=-1) + M_q)
        
        # Apply hard mask if provided (ground-truth DAG structure)
        # WARNING: This is applied AFTER softmax which may cause information leakage
        if hard_mask is not None:
            if is_multihead:
                if hard_mask.dim() == 2:
                    hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
                else:
                    hard_mask = hard_mask.unsqueeze(0)
            else:
                hard_mask = hard_mask.unsqueeze(0)
            
            att = att * hard_mask
        
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
            
        A = torch.nan_to_num(self.dropout(att))
        
        # Compute output values
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)
        
        return V.contiguous(), A, entropy


def build_causal_mask(p: torch.Tensor, n_heads: int = 1) -> torch.Tensor:
    """
    Args:
        p: (B, L, 1) tensor with the position of every token in the sequence.
        n_heads: number of attention heads

    Returns:
        For single head (n_heads=1): (B, L, L) mask M
        For multi-head (n_heads>1): (B, H, L, L) mask M
        with M[b, (h,) i, j] = -inf if p[b, j] > p[b, i], 0 otherwise.
    """
    p_flat = p.squeeze(-1)      # shape (B, L)

    p_i = p_flat.unsqueeze(-1)  # shape (B, L, 1)
    p_j = p_flat.unsqueeze(-2)  # shape (B, 1, L)

    # Build the additive mask (same dtype/device as `p`)
    M = torch.zeros_like(p_i.expand(-1, -1, p_flat.size(-1)))
    M.masked_fill_(p_j > p_i, float("-inf"))
    
    # Expand for multi-head if needed
    if n_heads > 1:
        M = M.unsqueeze(1).expand(-1, n_heads, -1, -1)  # (B, H, L, L)
    
    return M


def calculate_attention_entropy(att_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate entropy of attention weights.
    
    Args:
        att_weights: Attention weights tensor
                    - Multi-head: (B, H, L, S) 
                    - Single-head: (B, L, S)
        eps: Small value to avoid log(0)
    
    Returns:
        Entropy tensor:
        - Multi-head: (B, H, L) - entropy for each query position in each head
        - Single-head: (B, L) - entropy for each query position
    """
    # Clamp to avoid log(0)
    att_clamped = torch.clamp(att_weights, min=eps)
    
    # Calculate entropy: -sum(p * log(p)) along the key dimension (last dimension)
    log_att = torch.log(att_clamped)
    entropy = -torch.sum(att_weights * log_att, dim=-1)
    
    # Handle NaN values that might arise from 0 * log(0)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    return entropy


        
        
class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model_queries: int,
        d_model_keys: int,
        d_model_values: int,
        d_queries_keys: int,
        n_heads: int,
        mask_layer: nn.Module,
        attention_dropout: float,
        dropout_qkv: float,
        register_entropy: bool = False, 
        layer_name: str = None,
        query_seq_len: int = None,
        key_seq_len: int = None
        ):
        
        super(AttentionLayer, self).__init__()
        
        # Create DAGMask for attention types that support DAG learning
        from causaliT.core.modules.extra_layers import DAGMask
        
        # LieAttention, CausalCrossAttention, and PhiSoftMax all support DAG learning
        if attention in (LieAttention, CausalCrossAttention, PhiSoftMax) and query_seq_len is not None and key_seq_len is not None:
            mask_layer = DAGMask(n_heads=n_heads, query_seq_len=query_seq_len, key_seq_len=key_seq_len)
        
        self.inner_attention = attention(
            mask_layer=mask_layer,
            attention_dropout=attention_dropout,
            register_entropy=register_entropy,
            layer_name=layer_name
            )
        self.query_projection = nn.Linear(d_model_queries, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model_keys, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model_keys, d_model_values * n_heads)
        self.out_projection = nn.Linear(d_model_values * n_heads, d_model_queries)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        ):
        """
        Forward pass through attention layer.
        
        Args:
            query: Query tensor (B, L, d_model)
            key: Key tensor (B, S, d_model)
            value: Value tensor (B, S, d_model)
            mask_miss_k: Missing key mask
            mask_miss_q: Missing query mask
            pos: Positional encoding
            causal_mask: Whether to apply causal masking
            hard_mask: Optional hard mask tensor of shape (L, S) or (H, L, S).
                       Values in [0, 1], where 1 = attention allowed.
        """
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_heads
        
        # Apply projections and reshape for multi-head attention
        if H > 1:
            query = self.dropout_qkv(self.query_projection(query)).view(B, L, H, -1)
            key = self.dropout_qkv(self.key_projection(key)).view(B, S, H, -1)
            value = self.dropout_qkv(self.value_projection(value)).view(B, S, H, -1)
        else:
            query = self.dropout_qkv(self.query_projection(query)).view(B, L, -1)
            key = self.dropout_qkv(self.key_projection(key)).view(B, S, -1)
            value = self.dropout_qkv(self.value_projection(value)).view(B, S, -1)
            
        out, attn, ent = self.inner_attention(
            query=query,
            key=key,
            value=value,
            mask_miss_k=mask_miss_k,
            mask_miss_q=mask_miss_q,
            pos=pos,
            causal_mask=causal_mask,
            hard_mask=hard_mask,
            )
        
        # Reshape output and apply final projection if multi-head
        if H > 1:
            # out shape is (B, L, H, d_v) -> reshape to (B, L, H*d_v)
            out = out.view(B, L, -1)
            out = self.out_projection(out)
        else:
            # out shape is already (B, L, d_v)
            out = out.view(B, L, -1)
        
        return out, attn, ent
    
    
    
def main():
    """Quick test for both single-head and multi-head attention

    Returns:
        _tuple(torch.Tensor): attention output and score
    """
    
    bs = 1
    seq_len = 5
    d_model = 12
    d_queries_keys = 8
    mask = [False, True, True, False, False]
    x = torch.ones(bs,seq_len,d_model)
    x[0,0,0] = torch.nan
    
    print("Testing single-head attention (n_heads=1):")
    attention_single = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries = d_model,
        d_model_keys= d_model,
        d_model_values= d_model,
        d_queries_keys=d_queries_keys,
        n_heads=1,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0)
    
    out_single, score_single, ent = attention_single.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
        )
    
    print(f"Single-head - Output shape: {out_single.shape}, Score shape: {score_single.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    
    print("\nTesting multi-head attention (n_heads=4):")
    attention_multi = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries = d_model,
        d_model_keys= d_model,
        d_model_values= d_model,
        d_queries_keys=d_queries_keys,
        n_heads=4,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0)
    
    out_multi, score_multi, ent = attention_multi.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
        )
    
    print(f"Multi-head - Output shape: {out_multi.shape}, Score shape: {score_multi.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    
    # Test with causal mask and position
    print("\nTesting with causal mask:")
    pos = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).float()  # (1, 5, 1)
    
    out_causal, score_causal, ent = attention_multi.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=pos,
        causal_mask=True
        )
    
    print(f"Causal multi-head - Output shape: {out_causal.shape}, Score shape: {score_causal.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    
    # Test PhiSoftMax with DAG learning
    print("\n" + "="*60)
    print("Testing PhiSoftMax with learnable DAG mask:")
    print("="*60)
    
    attention_phi = AttentionLayer(
        attention=PhiSoftMax, 
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_queries_keys,
        n_heads=1,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0,
        query_seq_len=seq_len,
        key_seq_len=seq_len
    )
    
    out_phi, score_phi, ent_phi = attention_phi.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
    )
    
    print(f"PhiSoftMax single-head - Output shape: {out_phi.shape}, Score shape: {score_phi.shape}")
    print(f"  - phi shape: {attention_phi.inner_attention.phi.shape}")
    print(f"  - DAG probabilities shape: {attention_phi.inner_attention.get_dag_probabilities().shape}")
    
    # Test PhiSoftMax with hard_mask
    print("\nTesting PhiSoftMax with hard_mask:")
    hard_mask = torch.ones(seq_len, seq_len)
    hard_mask[0, 2] = 0  # Block attention from position 0 to position 2
    hard_mask[1, 3] = 0  # Block attention from position 1 to position 3
    
    out_phi_masked, score_phi_masked, _ = attention_phi.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False,
        hard_mask=hard_mask
    )
    
    print(f"  - Output shape with hard_mask: {out_phi_masked.shape}")
    print(f"  - Attention at masked positions [0,2]: {score_phi_masked[0, 0, 2].item():.6f} (should be ~0)")
    print(f"  - Attention at masked positions [1,3]: {score_phi_masked[0, 1, 3].item():.6f} (should be ~0)")
    
    # Test PhiSoftMax multi-head
    print("\nTesting PhiSoftMax multi-head (n_heads=4):")
    attention_phi_multi = AttentionLayer(
        attention=PhiSoftMax, 
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_queries_keys,
        n_heads=4,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0,
        query_seq_len=seq_len,
        key_seq_len=seq_len
    )
    
    out_phi_multi, score_phi_multi, ent_phi_multi = attention_phi_multi.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=pos,
        causal_mask=True
    )
    
    print(f"  - Output shape: {out_phi_multi.shape}")
    print(f"  - Score shape: {score_phi_multi.shape}")
    print(f"  - phi shape: {attention_phi_multi.inner_attention.phi.shape}")
    print(f"  - Entropy shape: {ent_phi_multi.shape if ent_phi_multi is not None else 'None'}")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
    
    
if __name__ == "__main__":
    main()
