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

from typing import Optional

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
        # Centroid-commit shadow (see causaliT/training/centroid_commit.py).
        # When enabled, ``embedding.weight`` holds the COMMITTED centroid (a
        # constant in the graph; written only by commit events) and ``shadow``
        # is a persistent buffer accumulating leaked gradient evidence.  The
        # forward reads the committed centroid; gradients land on the shadow
        # via a straight-through estimator.
        self.shadow: Optional[torch.Tensor]
        self.register_buffer("shadow", None, persistent=True)
        # Frozen rows (query parents prior, ``fixed: true``).  ``frozen_rows``
        # marks the table rows that must not move during training; a backward
        # hook zeroes their gradients and ``frozen_snapshot`` holds their
        # initialised values so ``reassert_frozen_rows`` can restore them after
        # every optimizer step (decoupled weight decay / structural gradient
        # noise would otherwise drift even zero-gradient rows).  Both buffers
        # are persistent so the freeze survives checkpoint save/load.
        self.register_buffer(
            "frozen_rows",
            torch.zeros(num_variables + var_id_offset, dtype=torch.bool),
            persistent=True,
        )
        self.frozen_snapshot: Optional[torch.Tensor]
        self.register_buffer("frozen_snapshot", None, persistent=True)
        self._freeze_hook_handle = None

    def enable_commit_shadow(self) -> None:
        """Register the evidence-accumulator shadow buffer (idempotent)."""
        if self.shadow is not None:
            return
        # Assignment (not register_buffer): ``shadow`` is already a registered
        # None buffer, so this keeps the persistent-buffer registration.
        self.shadow = self.embedding.weight.detach().clone()
        self.shadow.requires_grad_(True)

    def sync_shadow_to_weight(self) -> None:
        """Re-initialise the shadow at the committed weight (after lazy
        centroid init and after every commit event)."""
        if self.shadow is not None:
            with torch.no_grad():
                self.shadow.copy_(self.embedding.weight)

    def freeze_rows(self, indices) -> None:
        """Freeze the given rows for the rest of training (idempotent).

        Registers (once) a backward hook that zeroes the gradients of the
        frozen rows, and snapshots the CURRENT weight values so
        :meth:`reassert_frozen_rows` can restore them after each optimizer
        step.  Call again after writing new values into frozen rows (e.g. by
        the parents-prior init) to refresh the snapshot.
        """
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        if idx.numel() == 0:
            return
        with torch.no_grad():
            self.frozen_rows[idx] = True
            snap = self.embedding.weight.detach().clone()
            if self.frozen_snapshot is None:
                self.frozen_snapshot = snap
            else:
                self.frozen_snapshot.copy_(snap)
        if self._freeze_hook_handle is None:
            def _zero_frozen(grad):
                return grad.masked_fill(
                    self.frozen_rows.unsqueeze(1).to(grad.device), 0.0
                )
            self._freeze_hook_handle = self.embedding.weight.register_hook(
                _zero_frozen
            )

    def reassert_frozen_rows(self) -> None:
        """Restore frozen rows to their snapshot values (call after each
        optimizer step).  No-op when nothing is frozen."""
        if self.frozen_snapshot is None or not bool(self.frozen_rows.any()):
            return
        with torch.no_grad():
            self.embedding.weight[self.frozen_rows] = (
                self.frozen_snapshot[self.frozen_rows]
            )

    def _apply(self, fn, recurse=True):
        """Re-leaf the shadow buffer after device/dtype moves.

        ``nn.Module._apply`` (``.cuda()`` / ``.double()`` / ...) replaces
        buffer tensors with the op result, which is a NON-leaf tensor with
        ``requires_grad`` preserved.  A non-leaf shadow breaks ``.grad``
        accumulation (silently starving the commit evidence) and is rejected
        by ``deepcopy`` (cluster crash, job 12294402).  Re-detach it here.
        """
        out = super()._apply(fn, recurse)
        if self._buffers.get("shadow", None) is not None:
            self._buffers["shadow"] = (
                self._buffers["shadow"].detach().requires_grad_(True)
            )
        return out

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
        if self.shadow is None:
            return self.embedding(var_ids)
        # Straight-through: value = committed centroid, gradient -> shadow.
        committed = self.embedding(var_ids).detach()
        sh = self.shadow[var_ids]
        return committed + sh - sh.detach()

    def __repr__(self):
        return (f"FreeQueryEmbedding("
                f"num_variables={self.num_variables}, "
                f"d_model={self.d_model})")
