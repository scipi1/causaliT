"""
ValueIdentityEmbedding: vanilla-style learnable per-variable value embedding.

Purpose
-------
In the vanilla transformer benchmark (``comps_embed="summation"``) every token
is ``Linear(value) + nn.Embedding(variable_id)``: the node identity enters the
VALUE stream as a fully learnable, reconstruction-trained embedding SUMMED onto
the value map, before ``W_V``.

This module provides exactly that identity table for the SVFA value stream
(``value_structure_injection="learned_sum"`` in ``AttentionSelectorLayer``),
where the structural (Q/K) stream stays factorized but the value matches the
benchmark: ``V_j = W_V(v_j + e_j)``.

The table is a plain ``nn.Embedding`` configured IDENTICALLY to the vanilla
benchmark's variable embedding (see the ``nn_embedding`` module kwargs in the
benchmark configs): ``padding_idx=0``, ``sparse=False``, ``max_norm=1``.  It is
indexed by the 1-indexed variable-ID column of the input tensor (0 = padding).
"""

import torch
import torch.nn as nn


class ValueIdentityEmbedding(nn.Module):
    """
    Vanilla-style learnable per-variable identity embedding for the value stream.

    Maps a 1-indexed variable ID to a learnable ``d_model`` vector via a plain
    ``nn.Embedding`` with the same configuration as the vanilla benchmark's
    variable embedding (``padding_idx=0``, ``sparse=False``, ``max_norm=1``).

    Args:
        num_variables: Number of variables (e.g. ``X_seq_len``).
        d_model: Embedding dimension (spans the full d_model space).
        var_idx: Index of the variable-ID feature in the input tensor.
                 Defaults to 1 (production feature layout: value at col 0,
                 variable-ID at col 1), matching ``FreeQueryEmbedding``.
        var_id_offset: Variable IDs are 1-indexed in SCM datasets (0 = padding),
                       so the table has ``num_variables + var_id_offset`` rows
                       and is indexed by the raw (1-indexed) IDs.
        max_norm: ``nn.Embedding`` max_norm constraint (rows are renormalised
                  to this norm after every step); 1.0 matches the vanilla
                  benchmark's variable embedding.
        device: Target device (kept for API symmetry; the module is also moved
                by the parent module's ``.to(device)``).
    """

    def __init__(
        self,
        num_variables: int,
        d_model: int,
        var_idx: int = 1,
        var_id_offset: int = 1,
        max_norm: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model
        self.var_idx = var_idx
        self.var_id_offset = var_id_offset
        self.embedding = nn.Embedding(
            num_embeddings=num_variables + var_id_offset,
            embedding_dim=d_model,
            padding_idx=0,
            sparse=False,
            max_norm=max_norm,
            device=device,
            dtype=torch.float32,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor of shape (batch, seq_len, features).  Only the
               variable-ID column (``var_idx``) is used; the value column is
               ignored.

        Returns:
            Identity embeddings of shape (batch, seq_len, d_model).
        """
        var_ids = torch.nan_to_num(X[:, :, self.var_idx]).long()
        return self.embedding(var_ids)

    def __repr__(self):
        return (f"ValueIdentityEmbedding("
                f"num_variables={self.num_variables}, "
                f"d_model={self.d_model}, "
                f"max_norm={self.embedding.max_norm})")
