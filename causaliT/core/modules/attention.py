from os.path import dirname, abspath
import sys
from math import sqrt, log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causaliT.core.modules.extra_layers import (
    UniformAttentionMask, DAGLearningMixin,
    DAGMask, DAGMaskAntisym, DAGMaskGated
)
from causaliT.core.modules.orthogonal_linear import OrthogonalLinear
from causaliT.utils.entropy_utils import register_attention_entropy, calculate_attention_entropy
from typing import List, Optional


class LieAttention(DAGLearningMixin, nn.Module):
    """
    Lie Attention mechanism with DAG mask learning.
    
    Note: Acyclicity regularization (NOTEARS) only supports single-head attention.
          For multi-head attention, phi will have shape (H, L, S) and acyclicity 
          regularization should be disabled in the forecaster.
    
    The causal DAG is learned through a learnable tensor (phi) that models edge probabilities.
    A Gumbel-Softmax trick enables differentiable sampling from Bernoulli(sigmoid(phi)).
    Running averages of attention statistics are used as priors for KL regularization.
    
    Args:
        dag_mask: DAGMask, DAGMaskAntisym, DAGMaskGated, or None (passed from AttentionLayer)
        attention_dropout: Dropout rate for attention weights
        register_entropy: Whether to register entropy for logging
        layer_name: Name for logging purposes
    """
    def __init__(self, dag_mask: Optional[nn.Module], attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(LieAttention, self).__init__()
        
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
        # Initialize DAG learning via mixin (handles all DAGMask types)
        self._init_dag_learning(dag_mask)
        
        # --- Lie commutator amplification ---
        self.log_gain = nn.Parameter(torch.tensor(log(10.0)))   # start strong; set to 0.0 if you want gain=1
        self.log_tau_comm = nn.Parameter(torch.tensor(log(0.2)))  # tanh temperature (linear slope = gain/tau)
        self.max_gain = 1e3
        self.enforce_nonneg_flow = True  # set False if you want to allow negative flow
        
        
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
            # Get learnable Gumbel-Softmax temperature (clamped to safe range)
            tau_gs = torch.exp(self.log_tau_gs).clamp(self.tau_gs_min, self.tau_gs_max)
            
            # Sample batch DAG logits using Gumbel-Softmax trick
            u = torch.rand_like(self.phi)
            m_relaxed = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.phi) / tau_gs)
            M = m_relaxed
            
            # Zero out diagonal for antisymmetric DAG (no self-loops)
            # For DAGMaskAntisym, phi[i,i] = 0 leads to sigmoid(0) = 0.5
            # We explicitly force diagonal to 0 to ensure no self-loops
            from causaliT.core.modules.extra_layers import DAGMaskAntisym
            if isinstance(self.dag_mask, DAGMaskAntisym):
                if M.dim() == 2:
                    # Single-head: M shape is (L, S)
                    diag_mask = torch.eye(M.shape[-2], M.shape[-1], device=M.device, dtype=torch.bool)
                    M = M.masked_fill(diag_mask, 0.0)
                elif M.dim() == 3:
                    # Multi-head: M shape is (H, L, S)
                    diag_mask = torch.eye(M.shape[-2], M.shape[-1], device=M.device, dtype=torch.bool)
                    M = M.masked_fill(diag_mask.unsqueeze(0), 0.0)
            
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



class CausalCrossAttention(DAGLearningMixin, nn.Module):
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
    
    Args:
        dag_mask: DAGMask, DAGMaskAntisym, DAGMaskGated, or None (passed from AttentionLayer)
        attention_dropout: Dropout rate for attention weights
        register_entropy: Whether to register entropy for logging
        layer_name: Name for logging purposes
    """
    def __init__(self, dag_mask: Optional[nn.Module], attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(CausalCrossAttention, self).__init__()
        
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
        # Initialize DAG learning via mixin (handles all DAGMask types)
        self._init_dag_learning(dag_mask)
        
        # Gain/temperature parameters for GeLU(Tanh) activation
        # log_tau_act controls the sharpness of the tanh activation (not the DAG mask)
        self.log_gain = nn.Parameter(torch.tensor(log(1.0)))
        self.log_tau_act = nn.Parameter(torch.tensor(log(0.2)))  # tanh activation temperature
        self.max_gain = 10.0
    
    @property
    def phi(self) -> torch.Tensor:
        """Access phi through dag_mask to support both DAGMask and DAGMaskAntisym."""
        if self.dag_mask is not None:
            return self.dag_mask.phi
        return None
    
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
        tau_act = torch.exp(self.log_tau_act).clamp(1e-3, 10.0)
        
        # Causality-friendly activation: GeLU(Tanh(scores))
        att = F.gelu(F.tanh((gain / tau_act) * scores))
        
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
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)
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



class ToeplitzAttention(nn.Module):
    """
    Clean Toeplitz Attention for DAG learning.
    
    Decomposes attention scores into symmetric and antisymmetric parts:
        S = (QK^T + KQ^T) / 2   # Symmetric: edge existence probability
        A = (QK^T - KQ^T) / 2   # Antisymmetric: direction split
    
    Computes attention as:
        att[i,j] = sigmoid(S[i,j] / τ) × sigmoid(A[i,j] / τ)
    
    Properties:
        - att[i,j] + att[j,i] = sigmoid(S[i,j] / τ)  (budget constraint)
        - att[i,i] = 0                                 (no self-loops)
        - sigmoid(S[i,j]) = sigmoid(S[j,i])           (symmetric edge probability)
        - sigmoid(A[i,j]) + sigmoid(A[j,i]) = 1       (complementary direction split)
    
    The budget constraint is mathematically guaranteed:
        att[i,j] + att[j,i] = σ(S/τ) × σ(A/τ) + σ(S/τ) × σ(-A/τ)
                            = σ(S/τ) × [σ(A/τ) + σ(-A/τ)]
                            = σ(S/τ) × 1
                            = σ(S/τ) = P_edge[i,j]
    
    Temperature τ controls exploration vs exploitation:
        - High τ: Probabilities closer to 0.5 (exploration)
        - Low τ: Probabilities closer to 0 or 1 (exploitation)
    
    Args:
        dag_mask: Not used (kept for interface compatibility), should be None
        attention_dropout: Dropout rate for attention weights
        register_entropy: Whether to register entropy for logging
        layer_name: Name for logging purposes
        init_tau: Initial temperature (default: 1.0)
        tau_min: Minimum temperature (default: 0.1)
        tau_max: Maximum temperature (default: 5.0)
    """
    def __init__(
        self, 
        dag_mask: Optional[nn.Module], 
        attention_dropout: float, 
        register_entropy: bool, 
        layer_name: str,
        init_tau: float = 1.0,
        tau_min: float = 0.1,
        tau_max: float = 5.0,
    ):
        super(ToeplitzAttention, self).__init__()
        
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
        # dag_mask is not used - kept for interface compatibility
        self.dag_mask = None
        
        # Single temperature parameter for exploration → exploitation
        self.log_tau = nn.Parameter(torch.tensor(log(init_tau)))
        self.tau_min = tau_min
        self.tau_max = tau_max
    
    @property
    def tau(self) -> torch.Tensor:
        """Get the current temperature, clamped to valid range."""
        return torch.exp(self.log_tau).clamp(self.tau_min, self.tau_max)
    
    def get_dag_probabilities(self) -> torch.Tensor:
        """
        Returns the edge probabilities from the last forward pass.
        
        Note: This requires forward() to have been called first.
        Returns None if no forward pass has been computed yet.
        """
        if hasattr(self, 'last_att'):
            return self.last_att
        return None
    
    def get_edge_existence_probabilities(self) -> torch.Tensor:
        """
        Returns P_edge = sigmoid(S/τ), the symmetric edge existence probability.
        
        Note: This requires forward() to have been called first.
        """
        if hasattr(self, 'last_P_edge'):
            return self.last_P_edge
        return None
    
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
        Forward pass for Toeplitz Attention.
        
        Args:
            query: Query tensor (B, L, E) or (B, L, H, E) for multi-head
            key: Key tensor (B, S, E) or (B, S, H, E) for multi-head
            value: Value tensor (B, S, E) or (B, S, H, E) for multi-head
            mask_miss_k: Missing key mask (unused)
            mask_miss_q: Missing query mask (unused)
            pos: Positional encoding (unused - no causal masking in DAG attention)
            causal_mask: Whether to apply causal masking (unused)
            hard_mask: Optional hard mask tensor of shape (L, S) or (H, L, S).
                       Values in [0, 1], where 1 = attention allowed.
        """
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S_len, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S_len, _ = key.shape
            H = 1
        
        # Compute raw attention scores (no scaling - rely on normalization)
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        # Toeplitz decomposition
        S = (scores + scores.transpose(-1, -2)) / 2  # Symmetric: edge existence
        A = (scores - scores.transpose(-1, -2)) / 2  # Antisymmetric: direction
        
        # Get temperature
        tau = self.tau
        
        # Compute edge existence probability (symmetric)
        P_edge = torch.sigmoid(S / tau)  # P_edge[i,j] = P_edge[j,i]
        
        # Compute direction split (antisymmetric)
        d = torch.sigmoid(A / tau)  # d[i,j] + d[j,i] = 1
        
        # Final attention probabilities: att[i,j] = P_edge[i,j] × d[i,j]
        # Budget constraint: att[i,j] + att[j,i] = P_edge[i,j] (automatically satisfied!)
        att = P_edge * d
        
        # Zero out diagonal (no self-loops)
        if is_multihead:
            diag_mask = torch.eye(L, S_len, device=att.device, dtype=torch.bool)
            att = att.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), 0.0)
        else:
            diag_mask = torch.eye(L, S_len, device=att.device, dtype=torch.bool)
            att = att.masked_fill(diag_mask.unsqueeze(0), 0.0)
        
        # Apply hard mask if provided
        if hard_mask is not None:
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)
            att = att * hard_mask_expanded
        
        # Store for evaluation and inspection (detached, no gradients)
        self.last_att = att.detach().mean(dim=0)  # Average over batch
        self.last_P_edge = P_edge.detach().mean(dim=0)  # Average over batch
        self.last_S = S.detach().mean(dim=0)  # For analysis
        self.last_A = A.detach().mean(dim=0)  # For analysis
        
        # Store P_edge with gradients for L1 regularization in forecaster
        # (mean across batch dimension to get per-edge probability)
        # Usage: L1_loss = attention.inner_attention.P_edge_for_reg.mean()
        self.P_edge_for_reg = P_edge.mean(dim=0)  # (L, S) or (H, L, S)
        
        # Calculate entropy
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
        
        # Apply dropout
        A_out = self.dropout(att)
        A_out = torch.nan_to_num(A_out)
        
        # Compute output values
        if is_multihead:
            V_out = torch.einsum("bhls,bshd->blhd", A_out, value)
        else:
            V_out = torch.einsum("bls,bsd->bld", A_out, value)
        
        return V_out.contiguous(), A_out, entropy


class ToeplitzLieAttention(nn.Module):
    """
    Toeplitz-Lie Attention with symmetric gate + antisymmetric direction.
    
    This attention mechanism decomposes QK^T into symmetric and antisymmetric
    parts for DAG learning:
    
        S = (QK^T + KQ^T) / 2  # Symmetric: edge existence
        A = (QK^T - KQ^T) / 2  # Antisymmetric: flow direction (Lie commutator)
    
    Final edge probability:
        P(i→j) = σ(γ_ij) × σ(φ_ij)
    
    Where:
        γ_ij = gain_gate * tanh(S_ij / tau_gate) + γ_bias_ij  # symmetric gate
        φ_ij = gain_dir * tanh(A_ij / tau_dir) + φ_bias_ij    # antisymmetric direction
    
    Properties:
    - P(i→j) = 0 and P(j→i) = 0 is possible (when gate is closed)
    - P(i→j) + P(j→i) ≤ 1 (always, by construction)
    - P(i→i) = 0 (diagonal forced to 0)
    - Attention-derived: uses QK^T structure, not just learnable parameters
    - Optional learnable biases for fine-tuning
    
    See docs/TOEPLITZ_DECOMPOSITION.md for theoretical background.
    
    Args:
        dag_mask: DAGMask, DAGMaskAntisym, DAGMaskGated, or None
        attention_dropout: Dropout rate for attention weights
        register_entropy: Whether to register entropy for logging
        layer_name: Name for logging purposes
        init_gain_gate: Initial gain for symmetric gate (default: 2.0, was 5.0)
        init_gain_dir: Initial gain for direction (default: 3.0, was 10.0)
        init_tau_gate: Initial temperature for gate (default: 0.5)
        init_tau_dir: Initial temperature for direction (default: 0.3, was 0.2)
        max_gain: Maximum allowed gain during training (default: 20.0, was 100.0)
    """
    def __init__(
        self, 
        dag_mask: Optional[nn.Module], 
        attention_dropout: float, 
        register_entropy: bool, 
        layer_name: str,
        init_gain_gate: float = 2.0,
        init_gain_dir: float = 3.0,
        init_tau_gate: float = 0.5,
        init_tau_dir: float = 0.3,
        max_gain: float = 20.0,
    ):
        
        super(ToeplitzLieAttention, self).__init__()
        
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
        # Store dag_mask directly (can be DAGMask, DAGMaskAntisym, DAGMaskGated, or None)
        self.dag_mask = dag_mask
        
        # Symmetric gate parameters (for S = (QK^T + KQ^T) / 2)
        # Lower gains and higher temperatures lead to less decisive (more uncertain) probabilities
        self.log_gain_gate = nn.Parameter(torch.tensor(log(init_gain_gate)))
        self.log_tau_gate = nn.Parameter(torch.tensor(log(init_tau_gate)))
        self.max_gain = max_gain
        
        # Antisymmetric direction parameters (for A = (QK^T - KQ^T) / 2)
        self.log_gain_dir = nn.Parameter(torch.tensor(log(init_gain_dir)))
        self.log_tau_dir = nn.Parameter(torch.tensor(log(init_tau_dir)))
        
        # Gumbel-Softmax temperature for sampling
        self.log_tau_gs = nn.Parameter(torch.tensor(log(2.0)))
        self.tau_gs_min = 0.1
        self.tau_gs_max = 5.0
        
        # Whether to use learnable biases from dag_mask
        self.use_learnable_bias = True
    
    @property
    def phi(self) -> torch.Tensor:
        """Access phi (direction logits) through dag_mask if available."""
        if self.dag_mask is not None:
            return self.dag_mask.phi
        return None
    
    @property
    def gamma(self) -> torch.Tensor:
        """Access gamma (gate logits) through dag_mask if available (only DAGMaskGated has gamma)."""
        if self.dag_mask is not None and hasattr(self.dag_mask, 'gamma'):
            return self.dag_mask.gamma
        return None
    
    def get_dag_probabilities(self) -> torch.Tensor:
        """
        Returns P(i→j) = σ(γ_ij) × σ(φ_ij)
        
        Note: This returns the learnable bias-only DAG if no attention has been computed.
        For the attention-derived DAG, access self.last_dag_probs after forward().
        """
        # TODO check that diag is zero
        if self.dag_mask is not None:
            return self.dag_mask.get_dag_probabilities()
        return None
    
    def get_dag_logits(self) -> torch.Tensor:
        """Returns the antisymmetric direction logits (phi) for compatibility."""
        return self.phi
    
    def _compute_toeplitz_decomposition(self, scores: torch.Tensor) -> tuple:
        """
        Decompose attention scores into symmetric and antisymmetric parts.
        
        Args:
            scores: Raw QK^T scores, shape (B, L, S) or (B, H, L, S)
            
        Returns:
            tuple: (S_symmetric, A_antisymmetric)
        """
        # S = (QK^T + KQ^T) / 2
        S = (scores + scores.transpose(-1, -2)) / 2
        
        # A = (QK^T - KQ^T) / 2 (this is the Lie commutator / 2)
        A = (scores - scores.transpose(-1, -2)) / 2
        
        return S, A
    
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
        Forward pass for Toeplitz-Lie Attention.
        
        Args:
            query: Query tensor (B, L, E) or (B, L, H, E)
            key: Key tensor (B, S, E) or (B, S, H, E)
            value: Value tensor (B, S, E) or (B, S, H, E)
            mask_miss_k: Missing key mask (unused)
            mask_miss_q: Missing query mask (unused)
            pos: Positional encoding for causal masking
            causal_mask: Whether to apply causal masking
            hard_mask: Optional hard mask tensor
        """
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S_len, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S_len, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute raw attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        # Scale scores
        scores = scale * scores
        
        # Toeplitz decomposition
        S_sym, A_antisym = self._compute_toeplitz_decomposition(scores)
        
        # Get gains and temperatures
        gain_gate = torch.exp(self.log_gain_gate).clamp(1e-3, self.max_gain)
        tau_gate = torch.exp(self.log_tau_gate).clamp(1e-3, 10.0)
        gain_dir = torch.exp(self.log_gain_dir).clamp(1e-3, self.max_gain)
        tau_dir = torch.exp(self.log_tau_dir).clamp(1e-3, 10.0)
        
        # Compute gate logits from symmetric part
        gamma_att = gain_gate * torch.tanh(S_sym / tau_gate)  # (B, *, L, S)
        
        # Compute direction logits from antisymmetric part
        phi_att = gain_dir * torch.tanh(A_antisym / tau_dir)  # (B, *, L, S)
        
        # Add learnable biases if available
        if self.dag_mask is not None and self.use_learnable_bias:
            # Only add gamma bias if dag_mask has gamma (only DAGMaskGated has gamma)
            gamma_bias = self.gamma  # (L, S) or (H, L, S), or None
            if gamma_bias is not None:
                gamma_att = gamma_att + gamma_bias.unsqueeze(0)
            
            # Only add phi bias if dag_mask has phi (DAGMask, DAGMaskAntisym, DAGMaskGated all have phi)
            phi_bias = self.phi  # (L, S) or (H, L, S), or None
            if phi_bias is not None:
                phi_att = phi_att + phi_bias.unsqueeze(0)
        
        # Compute DAG probabilities: P(i→j) = σ(γ) × σ(φ)
        gate_probs = torch.sigmoid(gamma_att)
        dir_probs = torch.sigmoid(phi_att)
        dag_probs = gate_probs * dir_probs
        
        # Zero out diagonal (no self-loops)
        if is_multihead:
            diag_mask = torch.eye(L, S_len, device=dag_probs.device, dtype=torch.bool)
            dag_probs = dag_probs.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), 0.0)
        else:
            diag_mask = torch.eye(L, S_len, device=dag_probs.device, dtype=torch.bool)
            dag_probs = dag_probs.masked_fill(diag_mask.unsqueeze(0), 0.0)
        
        # Store for inspection/evaluation and regularization
        self.last_dag_probs = dag_probs.detach()
        self.last_gate_probs = gate_probs.detach()  # For L1 regularization on symmetric gate
        self.last_S_sym = S_sym.detach()  # Symmetric part for analysis
        
        # Store gate_probs with gradients for L1 regularization in forecaster
        # (mean across batch dimension to get per-edge gate probability)
        self.gate_probs_for_reg = gate_probs.mean(dim=0)  # (L, S) or (H, L, S)
        
        # Apply Gumbel-Softmax for differentiable sampling during training
        tau_gs = torch.exp(self.log_tau_gs).clamp(self.tau_gs_min, self.tau_gs_max)
        if self.training:
            # Gumbel noise for gate
            u_gate = torch.rand_like(gamma_att)
            gumbel_gate = torch.log(u_gate + 1e-8) - torch.log(1 - u_gate + 1e-8)
            gate_sample = torch.sigmoid((gumbel_gate + gamma_att) / tau_gs)
            
            # Gumbel noise for direction
            u_dir = torch.rand_like(phi_att)
            gumbel_dir = torch.log(u_dir + 1e-8) - torch.log(1 - u_dir + 1e-8)
            dir_sample = torch.sigmoid((gumbel_dir + phi_att) / tau_gs)
            
            M = gate_sample * dir_sample
        else:
            M = dag_probs
        
        # Zero out diagonal in mask
        if is_multihead:
            M = M.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), 0.0)
        else:
            M = M.masked_fill(diag_mask.unsqueeze(0), 0.0)
        
        # Compute ReLU attention from direction (Lie-style)
        # Comment: why ReLU? 
        # ReLu(1) = 1 --> good probabilistic limit, GeLU(1)= 0.84
        att = F.relu(torch.tanh(A_antisym * gain_dir / tau_dir))
        
        # Apply DAG mask
        att = att * M
        
        # Apply hard mask if provided
        if hard_mask is not None:
            if is_multihead:
                if hard_mask.dim() == 2:
                    hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
                else:
                    hard_mask = hard_mask.unsqueeze(0)
            else:
                hard_mask = hard_mask.unsqueeze(0)
            att = att * hard_mask
        
        # Calculate entropy
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
        
        # Apply dropout
        A = self.dropout(att)
        A = torch.nan_to_num(A)
        
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
    def __init__(self, dag_mask: Optional[nn.Module], attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(PhiSoftMax, self).__init__()
        
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
        # Store dag_mask directly
        self.dag_mask = dag_mask
        if dag_mask is not None:
            phi_value = dag_mask.phi
            self.register_buffer('runav_att_mean', torch.zeros_like(phi_value))
            self.register_buffer('runav_att_snr', torch.zeros_like(phi_value))
        else:
            self.runav_att_mean = None
            self.runav_att_snr = None
        
        # Gumbel-Softmax temperature - learnable with annealing
        # Starts high (τ=2.0) for exploration, anneals toward low values for sharper masks
        # Named log_tau_gs to be consistent with other attention types
        self.log_tau_gs = nn.Parameter(torch.tensor(log(2.0)))
        self.tau_gs_min = 0.1  # Minimum temperature
        self.tau_gs_max = 5.0  # Maximum temperature
        
        # Threshold for converting soft phi to hard mask (values below threshold -> -inf)
        # This is learnable to allow the model to adjust the sparsity
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.threshold_min = 0.01
        self.threshold_max = 0.99
        
        # Large constant for soft thresholding (makes the transition sharp but differentiable)
        self.mask_scale = 1e4
    
    @property
    def phi(self) -> torch.Tensor:
        """Access phi through dag_mask to support both DAGMask and DAGMaskAntisym."""
        if self.dag_mask is not None:
            return self.dag_mask.phi
        return None
    
    def get_dag_probabilities(self) -> torch.Tensor:
        """
        Returns the posterior probability of each edge being active in the learned DAG.
        
        This is useful for:
        - Inference: Extract the learned causal structure
        - Visualization: Plot the learned DAG
        - Evaluation: Compare against ground-truth DAG
        
        For DAGMaskAntisym, the diagonal is forced to 0 (no self-loops) since
        sigmoid(0) = 0.5 would otherwise appear on the diagonal.
        
        Returns:
            torch.Tensor: Edge probabilities in [0, 1], shape (L, S) or (H, L, S) for multi-head.
                         Returns None if no DAG is being learned (phi is None).
                         Diagonal is 0 for antisymmetric parameterization.
        """
        if self.phi is not None:
            probs = torch.sigmoid(self.phi)
            
            # For antisymmetric DAG, zero out diagonal (no self-loops)
            from causaliT.core.modules.extra_layers import DAGMaskAntisym
            if isinstance(self.dag_mask, DAGMaskAntisym):
                if probs.dim() == 2:
                    diag_mask = torch.eye(probs.shape[-2], probs.shape[-1], device=probs.device, dtype=torch.bool)
                    probs = probs.masked_fill(diag_mask, 0.0)
                elif probs.dim() == 3:
                    diag_mask = torch.eye(probs.shape[-2], probs.shape[-1], device=probs.device, dtype=torch.bool)
                    probs = probs.masked_fill(diag_mask.unsqueeze(0), 0.0)
            
            return probs
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
            # Get learnable Gumbel-Softmax temperature (clamped to safe range)
            tau_gs = torch.exp(self.log_tau_gs).clamp(self.tau_gs_min, self.tau_gs_max)
            
            # Sample batch DAG logits using Gumbel-Softmax trick
            u = torch.rand_like(self.phi)
            gumbel_noise = torch.log(u + 1e-8) - torch.log(1 - u + 1e-8)
            m_relaxed = torch.sigmoid((gumbel_noise + self.phi) / tau_gs)
            
            # Zero out diagonal for antisymmetric DAG (no self-loops)
            # For DAGMaskAntisym, phi[i,i] = 0 leads to sigmoid(0) = 0.5
            # We explicitly force diagonal to 0 to ensure no self-loops
            from causaliT.core.modules.extra_layers import DAGMaskAntisym
            if isinstance(self.dag_mask, DAGMaskAntisym):
                if m_relaxed.dim() == 2:
                    # Single-head: m_relaxed shape is (L, S)
                    diag_mask = torch.eye(m_relaxed.shape[-2], m_relaxed.shape[-1], device=m_relaxed.device, dtype=torch.bool)
                    m_relaxed = m_relaxed.masked_fill(diag_mask, 0.0)
                elif m_relaxed.dim() == 3:
                    # Multi-head: m_relaxed shape is (H, L, S)
                    diag_mask = torch.eye(m_relaxed.shape[-2], m_relaxed.shape[-1], device=m_relaxed.device, dtype=torch.bool)
                    m_relaxed = m_relaxed.masked_fill(diag_mask.unsqueeze(0), 0.0)
            
            # Clamp threshold to valid range
            thresh = self.threshold.clamp(self.threshold_min, self.threshold_max)
            
            # Convert to additive mask using soft thresholding (differentiable)
            # Values below threshold get large negative values, above threshold get ~0
            # Using: -scale * relu(threshold - m_relaxed) 
            # This gives: 0 when m_relaxed >= threshold, negative when m_relaxed < threshold
            phi_mask_additive = -self.mask_scale * F.relu(thresh - m_relaxed)
            
            # For antisymmetric DAG, force diagonal to -inf (no self-attention)
            if isinstance(self.dag_mask, DAGMaskAntisym):
                if phi_mask_additive.dim() == 2:
                    diag_mask = torch.eye(phi_mask_additive.shape[-2], phi_mask_additive.shape[-1], device=phi_mask_additive.device, dtype=torch.bool)
                    phi_mask_additive = phi_mask_additive.masked_fill(diag_mask, -self.mask_scale)
                elif phi_mask_additive.dim() == 3:
                    diag_mask = torch.eye(phi_mask_additive.shape[-2], phi_mask_additive.shape[-1], device=phi_mask_additive.device, dtype=torch.bool)
                    phi_mask_additive = phi_mask_additive.masked_fill(diag_mask.unsqueeze(0), -self.mask_scale)
            
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
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)
            
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
    
    Note: This class does not use DAG learning, but accepts dag_mask parameter
    for interface consistency with AttentionLayer.
    """
    def __init__(self, dag_mask: Optional[nn.Module], attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(ScaledDotAttention, self).__init__()
        
        self.dag_mask = dag_mask  # Not used, but kept for interface consistency
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
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)
            
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
    
    Note: This class does not use DAG learning, but accepts dag_mask parameter
    for interface consistency with AttentionLayer.
    """
    def __init__(self, dag_mask: Optional[nn.Module], attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(ScaledDotAttentionNAIM, self).__init__()
        
        self.dag_mask = dag_mask  # Not used, but kept for interface consistency
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


def expand_hard_mask(hard_mask: torch.Tensor, is_multihead: bool, batch_size: int) -> torch.Tensor:
    """
    Expand hard_mask to match attention scores shape.
    
    Handles both static masks (no batch dim) and in-context masks (with batch dim).
    This is a centralized utility function used by all attention classes.
    
    Args:
        hard_mask: Mask tensor. Can be:
            - (L, S): static mask for all samples and heads
            - (H, L, S): per-head static mask (multihead only)
            - (B, L, S): per-sample in-context mask
            - (B, H, L, S): per-sample per-head mask (multihead only)
        is_multihead: Whether attention is multi-head
        batch_size: Batch size to detect if first dim is batch or heads
        
    Returns:
        Expanded mask matching scores shape: (B, L, S) or (B, H, L, S)
    """
    if is_multihead:
        if hard_mask.dim() == 2:
            # (L, S) -> (1, 1, L, S) - static mask for all samples/heads
            hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
        elif hard_mask.dim() == 3:
            # Could be (H, L, S) or (B, L, S) - check first dim
            if hard_mask.shape[0] == batch_size:
                # (B, L, S) -> (B, 1, L, S) - in-context mask, broadcast to all heads
                hard_mask = hard_mask.unsqueeze(1)
            else:
                # (H, L, S) -> (1, H, L, S) - per-head static mask
                hard_mask = hard_mask.unsqueeze(0)
        # elif dim == 4: already (B, H, L, S) - no change needed
    else:
        if hard_mask.dim() == 2:
            # (L, S) -> (1, L, S) - static mask
            hard_mask = hard_mask.unsqueeze(0)
        # elif dim == 3: already (B, L, S) - no change needed (in-context mask)
    return hard_mask


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
    """
    Multi-head attention layer with centralized DAG mask creation.
    
    This layer centralizes DAGMask creation and passes dag_mask (or None) directly 
    to the inner attention module, eliminating the need for isinstance checks 
    in each attention class.
    
    For SVFA (Structure-Value Factorized Attention), the caller (encoder/decoder)
    should pass the appropriate tensors:
        - query, key: Structure embeddings (for Q, K projections)
        - value: Value embeddings (for V projection)
    
    Args:
        attention: Attention class (LieAttention, CausalCrossAttention, PhiSoftMax, etc.)
        dag_parameterization: str, one of:
            - "independent": Each edge (i,j) is independent. Allows bidirectional edges. (default)
            - "antisymmetric": P(i→j) + P(j→i) = 1. Requires square attention (self-attention only).
            - "gated": Symmetric gate + antisymmetric direction. Requires square attention.
        key_projection_type: str, one of:
            - "linear": Standard unconstrained linear projection (default)
            - "orthogonal": Orthogonal projection (rotation + optional scaling)
              Preserves inner products: if inputs are orthogonal, outputs remain orthogonal.
              Useful when using orthogonal embeddings (e.g., OrthogonalMaskEmbedding).
              Requires d_queries_keys >= d_model_keys.
        orthogonal_scale: Whether to include learnable scale in orthogonal projection (default: True)
        orthogonal_init_scale: Initial scale value for orthogonal projection (default: 1.0)
        toeplitz_init_gain_gate: Initial gain for ToeplitzLieAttention symmetric gate (default: 2.0)
        toeplitz_init_gain_dir: Initial gain for ToeplitzLieAttention direction (default: 3.0)
        toeplitz_init_tau_gate: Initial temperature for ToeplitzLieAttention gate (default: 0.5)
        toeplitz_init_tau_dir: Initial temperature for ToeplitzLieAttention direction (default: 0.3)
        toeplitz_max_gain: Maximum allowed gain for ToeplitzLieAttention (default: 20.0)
    """
    def __init__(
        self,
        attention: nn.Module,
        d_model_queries: int,
        d_model_keys: int,
        d_model_values: int,
        d_queries_keys: int,
        n_heads: int,
        mask_layer: nn.Module,  # Legacy parameter, may be ignored for DAG-learning attention
        attention_dropout: float,
        dropout_qkv: float,
        register_entropy: bool = False, 
        layer_name: str = None,
        query_seq_len: int = None,
        key_seq_len: int = None,
        dag_parameterization: str = "independent",
        phi_init_std: float = 0.1,
        # Key projection type for preserving orthogonality
        key_projection_type: str = "linear",
        orthogonal_scale: bool = True,
        orthogonal_init_scale: float = 1.0,
        # ToeplitzLieAttention specific parameters (configurable)
        toeplitz_init_gain_gate: float = 2.0,
        toeplitz_init_gain_dir: float = 3.0,
        toeplitz_init_tau_gate: float = 0.5,
        toeplitz_init_tau_dir: float = 0.3,
        toeplitz_max_gain: float = 20.0,
        ):
        
        super(AttentionLayer, self).__init__()
        
        # Attention types that require square attention (self-attention only)
        SELF_ATTENTION_ONLY = (LieAttention, ToeplitzLieAttention, ToeplitzAttention)
        
        # Check square attention requirement for self-attention-only modules
        if attention in SELF_ATTENTION_ONLY and query_seq_len is not None and key_seq_len is not None:
            if query_seq_len != key_seq_len:
                raise ValueError(
                    f"{attention.__name__} requires square attention (self-attention) but got "
                    f"query_seq_len={query_seq_len} != key_seq_len={key_seq_len}. "
                    f"Use CausalCrossAttention or ScaledDotAttention for cross-attention instead."
                )
        
        # Attention types that support DAG learning
        DAG_LEARNING_ATTENTION = (LieAttention, CausalCrossAttention, PhiSoftMax, ToeplitzLieAttention)
        
        # Centralized DAGMask creation
        dag_mask = None
        if attention in DAG_LEARNING_ATTENTION and query_seq_len is not None and key_seq_len is not None:
            dag_mask = self._create_dag_mask(
                dag_parameterization=dag_parameterization,
                n_heads=n_heads,
                query_seq_len=query_seq_len,
                key_seq_len=key_seq_len,
                phi_init_std=phi_init_std
            )
        
        # Create inner attention - pass dag_mask and attention-specific parameters
        if attention == ToeplitzLieAttention:
            self.inner_attention = attention(
                dag_mask=dag_mask,
                attention_dropout=attention_dropout,
                register_entropy=register_entropy,
                layer_name=layer_name,
                init_gain_gate=toeplitz_init_gain_gate,
                init_gain_dir=toeplitz_init_gain_dir,
                init_tau_gate=toeplitz_init_tau_gate,
                init_tau_dir=toeplitz_init_tau_dir,
                max_gain=toeplitz_max_gain,
            )
        elif attention == ToeplitzAttention:
            # ToeplitzAttention: Clean Toeplitz decomposition with single temperature
            # Does not use dag_mask - derives DAG directly from attention scores
            self.inner_attention = attention(
                dag_mask=None,  # Not used
                attention_dropout=attention_dropout,
                register_entropy=register_entropy,
                layer_name=layer_name,
                init_tau=toeplitz_init_tau_gate,  # Reuse gate tau as the single temperature
                tau_min=0.1,
                tau_max=5.0,
            )
        else:
            self.inner_attention = attention(
                dag_mask=dag_mask,
                attention_dropout=attention_dropout,
                register_entropy=register_entropy,
                layer_name=layer_name
            )
        
        # Projection layers
        self.query_projection = nn.Linear(d_model_queries, d_queries_keys * n_heads)
        
        # Key projection: supports orthogonal projection for preserving embedding orthogonality
        self.key_projection_type = key_projection_type
        if key_projection_type == "linear":
            self.key_projection = nn.Linear(d_model_keys, d_queries_keys * n_heads)
        elif key_projection_type == "orthogonal":
            # Orthogonal projection preserves inner products: ⟨W_k x, W_k y⟩ = ⟨x, y⟩
            # Requires d_queries_keys >= d_model_keys (more output dims than input)
            # For multi-head, we need d_queries_keys * n_heads >= d_model_keys
            out_features = d_queries_keys * n_heads
            in_features = d_model_keys
            if out_features < in_features:
                raise ValueError(
                    f"Orthogonal key projection requires d_queries_keys * n_heads >= d_model_keys, "
                    f"got {out_features} < {in_features}. "
                    f"Either increase d_queries_keys or use key_projection_type='linear'."
                )
            self.key_projection = OrthogonalLinear(
                in_features=in_features,
                out_features=out_features,
                use_scale=orthogonal_scale,
                init_scale=orthogonal_init_scale
            )
        else:
            raise ValueError(
                f"Invalid key_projection_type='{key_projection_type}'. "
                f"Must be one of: 'linear', 'orthogonal'"
            )
        
        self.value_projection = nn.Linear(d_model_values, d_model_values * n_heads)
        self.out_projection = nn.Linear(d_model_values * n_heads, d_model_values)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads
    
    def _create_dag_mask(
        self,
        dag_parameterization: str,
        n_heads: int,
        query_seq_len: int,
        key_seq_len: int,
        phi_init_std: float
    ) -> Optional[nn.Module]:
        """
        Create the appropriate DAGMask based on parameterization and attention shape.
        
        Args:
            dag_parameterization: One of "independent", "antisymmetric", "gated", or None
            n_heads: Number of attention heads
            query_seq_len: Length of query sequence
            key_seq_len: Length of key sequence
            phi_init_std: Standard deviation for phi initialization
            
        Returns:
            DAGMask, DAGMaskAntisym, DAGMaskGated, or None (if dag_parameterization is None)
        """
        # Handle "no learnable phi" case - return None (no DAGMask)
        if dag_parameterization is None:
            return None
        
        is_square = (query_seq_len == key_seq_len)
        
        # Validate parameterization
        valid_parameterizations = ("independent", "antisymmetric", "gated")
        if dag_parameterization not in valid_parameterizations:
            raise ValueError(
                f"Invalid dag_parameterization='{dag_parameterization}'. "
                f"Must be one of: {valid_parameterizations} or None"
            )
        
        if dag_parameterization == "gated":
            # Gated: symmetric gate + antisymmetric direction (requires square attention)
            if not is_square:
                raise ValueError(
                    f"dag_parameterization='gated' requires square attention (self-attention) but got "
                    f"query_seq_len={query_seq_len} != key_seq_len={key_seq_len}. "
                    f"For cross-attention, use dag_parameterization='independent' instead."
                )
            return DAGMaskGated(
                n_heads=n_heads, 
                query_seq_len=query_seq_len, 
                key_seq_len=key_seq_len,
                init_std_gate=0.0,
                init_std_dir=phi_init_std
            )
        
        elif dag_parameterization == "antisymmetric":
            # Antisymmetric: P(i→j) + P(j→i) = 1 (requires square attention)
            if not is_square:
                raise ValueError(
                    f"dag_parameterization='antisymmetric' requires square attention (self-attention) but got "
                    f"query_seq_len={query_seq_len} != key_seq_len={key_seq_len}. "
                    f"For cross-attention, use dag_parameterization='independent' instead."
                )
            return DAGMaskAntisym(n_heads=n_heads, query_seq_len=query_seq_len, key_seq_len=key_seq_len, init_std=phi_init_std)
        
        else:
            # Default: independent parameterization (works for any shape)
            return DAGMask(n_heads=n_heads, query_seq_len=query_seq_len, key_seq_len=key_seq_len, init_std=phi_init_std)

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
            query: Query tensor (B, L, d_model) - for Q projection
            key: Key tensor (B, S, d_model) - for K projection
            value: Value tensor (B, S, d_model) - for V projection
            mask_miss_k: Missing key mask
            mask_miss_q: Missing query mask
            pos: Positional encoding
            causal_mask: Whether to apply causal masking
            hard_mask: Optional hard mask tensor of shape (L, S) or (H, L, S).
                       Values in [0, 1], where 1 = attention allowed.
                       
        Note for SVFA:
            The caller should pass structure embeddings for query/key and
            value embeddings for value. This layer is agnostic to factorization.
        """
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_heads
        
        # Apply projections and reshape for multi-head attention
        if H > 1:
            q = self.dropout_qkv(self.query_projection(query)).view(B, L, H, -1)
            k = self.dropout_qkv(self.key_projection(key)).view(B, S, H, -1)
            v = self.dropout_qkv(self.value_projection(value)).view(B, S, H, -1)
        else:
            q = self.dropout_qkv(self.query_projection(query)).view(B, L, -1)
            k = self.dropout_qkv(self.key_projection(key)).view(B, S, -1)
            v = self.dropout_qkv(self.value_projection(value)).view(B, S, -1)
            
        out, attn, ent = self.inner_attention(
            query=q,
            key=k,
            value=v,
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
    
    # Test orthogonal key projection
    print("\n" + "="*60)
    print("Testing Orthogonal Key Projection:")
    print("="*60)
    
    # Create orthogonal input embeddings (simulate OrthogonalMaskEmbedding output)
    d_model_orth = 12  # Divisible by 3 variables
    d_qk = 16  # d_qk >= d_model for orthogonal projection
    
    attention_orth = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries=d_model_orth,
        d_model_keys=d_model_orth,
        d_model_values=d_model_orth,
        d_queries_keys=d_qk,
        n_heads=1,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0,
        key_projection_type="orthogonal",  # Use orthogonal projection
        orthogonal_scale=True,
        orthogonal_init_scale=1.0
    )
    
    print(f"  Key projection type: {attention_orth.key_projection_type}")
    print(f"  Key projection: {attention_orth.key_projection}")
    print(f"  Orthonormal columns: {attention_orth.key_projection.verify_orthonormality()}")
    
    # Create orthogonal input embeddings (3 variables, 4 dims each)
    x_orth = torch.zeros(bs, 3, d_model_orth)
    x_orth[0, 0, :4] = torch.randn(4)   # Variable 1: dims 0-3
    x_orth[0, 1, 4:8] = torch.randn(4)  # Variable 2: dims 4-7
    x_orth[0, 2, 8:] = torch.randn(4)   # Variable 3: dims 8-11
    
    # Check input orthogonality
    input_dot_01 = torch.dot(x_orth[0, 0], x_orth[0, 1]).item()
    input_dot_02 = torch.dot(x_orth[0, 0], x_orth[0, 2]).item()
    input_dot_12 = torch.dot(x_orth[0, 1], x_orth[0, 2]).item()
    print(f"\n  Input dot products (should all be 0):")
    print(f"    ⟨x[0], x[1]⟩ = {input_dot_01:.6f}")
    print(f"    ⟨x[0], x[2]⟩ = {input_dot_02:.6f}")
    print(f"    ⟨x[1], x[2]⟩ = {input_dot_12:.6f}")
    
    # Project keys through orthogonal projection
    k_projected = attention_orth.key_projection(x_orth)
    
    # Check output orthogonality (should be preserved!)
    output_dot_01 = torch.dot(k_projected[0, 0], k_projected[0, 1]).item()
    output_dot_02 = torch.dot(k_projected[0, 0], k_projected[0, 2]).item()
    output_dot_12 = torch.dot(k_projected[0, 1], k_projected[0, 2]).item()
    print(f"\n  Output dot products (should all be ~0 for orthogonal projection):")
    print(f"    ⟨k[0], k[1]⟩ = {output_dot_01:.6f}")
    print(f"    ⟨k[0], k[2]⟩ = {output_dot_02:.6f}")
    print(f"    ⟨k[1], k[2]⟩ = {output_dot_12:.6f}")
    
    # Compare with linear projection (orthogonality NOT preserved)
    attention_linear = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries=d_model_orth,
        d_model_keys=d_model_orth,
        d_model_values=d_model_orth,
        d_queries_keys=d_qk,
        n_heads=1,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0,
        key_projection_type="linear"  # Default linear projection
    )
    
    k_linear = attention_linear.key_projection(x_orth)
    linear_dot_01 = torch.dot(k_linear[0, 0], k_linear[0, 1]).item()
    linear_dot_02 = torch.dot(k_linear[0, 0], k_linear[0, 2]).item()
    linear_dot_12 = torch.dot(k_linear[0, 1], k_linear[0, 2]).item()
    print(f"\n  Linear projection output dot products (likely NOT 0):")
    print(f"    ⟨k[0], k[1]⟩ = {linear_dot_01:.6f}")
    print(f"    ⟨k[0], k[2]⟩ = {linear_dot_02:.6f}")
    print(f"    ⟨k[1], k[2]⟩ = {linear_dot_12:.6f}")
    
    # Run full forward pass with orthogonal key projection
    out_orth, score_orth, ent_orth = attention_orth.forward(
        query=x_orth, 
        key=x_orth, 
        value=x_orth,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
    )
    print(f"\n  Full forward pass - Output shape: {out_orth.shape}")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
    
    
if __name__ == "__main__":
    main()
