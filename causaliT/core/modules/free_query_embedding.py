"""
FreeQueryEmbedding: unconstrained learnable per-variable query embedding.

Purpose
-------
In ``AttentionSelectorLayer`` the predicted (X) nodes are used in two roles:

* as attention **keys** — X_i is offered as a candidate parent to other X_j;
* as attention **queries** — X_i selects its own parents.

When a single embedding serves both roles, a gradient that updates
"X_i-as-child" (query) also perturbs "X_i-as-parent" (key), so the model cannot
learn ``X_i ← S`` and ``X_i ← X_j`` independently.  Giving the query its own
embedding removes that coupling.

Because the query is built from ``x_blanked`` (value column zeroed), only the
variable identity matters, so a plain lookup table (var_id → d_model vector) is
sufficient — there is no value pathway.  Unlike the *keys*, the queries do NOT
need to be mutually orthogonal, so this embedding is left fully free
(unconstrained) to maximise its ability to point at any key.
"""

import torch
import torch.nn as nn


class FreeQueryEmbedding(nn.Module):
    """
    Free (unconstrained) learnable per-variable identity embedding.

    Maps a 1-indexed variable ID to a learnable ``d_model`` vector via
    ``nn.Embedding`` (index 0 reserved for padding).

    Args:
        num_variables: Number of X variables (e.g. ``X_seq_len``).
        d_model: Embedding dimension (spans the full d_model space).
        var_idx: Index of the variable-ID feature in the input tensor.
                 Defaults to 1 to match ``OrthogonalMaskEmbedding``.
        var_id_offset: Variable IDs are 1-indexed in SCM datasets (0 = padding),
                       so the table has ``num_variables + var_id_offset`` rows and
                       is indexed by the raw (1-indexed) IDs.
        device: Target device (kept for API symmetry; ``nn.Embedding`` is moved
                by the parent module's ``.to(device)``).
    """

    def __init__(
        self,
        num_variables: int,
        d_model: int,
        var_idx: int = 1,
        var_id_offset: int = 1,
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
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor of shape (batch, seq_len, features).  Only the
               variable-ID column (``var_idx``) is used; the value column is
               ignored (it is blanked for queries anyway).

        Returns:
            Query identity embeddings of shape (batch, seq_len, d_model).
        """
        var_ids = torch.nan_to_num(X[:, :, self.var_idx]).long()
        return self.embedding(var_ids)

    def __repr__(self):
        return (f"FreeQueryEmbedding("
                f"num_variables={self.num_variables}, "
                f"d_model={self.d_model})")
