# Standard library imports
import logging

# Third-party imports
import torch
from pytorch_lightning import Callback, Trainer, LightningModule

logger = logging.getLogger(__name__)


class GradientLogger(Callback):
    """
    Logs ‖∇θ‖₂ and variance layer‑by‑layer.
    Optionally stores raw gradients as .pt files.
    """
    def __init__(self):
        super().__init__()        

    @staticmethod
    def _stats(t: torch.Tensor):
        return dict(
            grad_norm = t.norm().item(),
        )

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        self.metrics = {}
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            s = self._stats(p.grad.detach())
            self.metrics[f"grad_norm/{name}"] = s["grad_norm"]
    
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log_dict(
                self.metrics,
                on_step=False,
                on_epoch=True
            )
        

class PeriodicDAGMetrics(Callback):
    """Log structure-quality trajectory metrics from the live attention posterior.

    WHY NOT phi / evaluate_dag_from_model: on this architecture the explicit DAG
    parameterisation is DEPRECATED (``phi`` / ``dag_mask`` are gone) and its
    documented fallback reads ``batch_att_mean``, which is never assigned any
    more (zero occurrences repo-wide).  That path therefore returns None at
    EVERY epoch, not merely before the first forward pass.  The attention scores
    arise during the forward pass from the structural embeddings, so the
    batch-mean posterior stashed by ``_step`` (``_last_att_mean``) is the only
    valid soft adjacency -- and it is free, needing no extra forward pass.

    WHY NOT SHD/TPR/FDR here: those need a hard 0.5 cut of a min-max rescaled
    matrix, which manufactures structure out of near-uniform attention and is
    not comparable across epochs.  The end-of-run ``eval_attention_scores``
    already reports them properly.  This callback logs THRESHOLD-FREE
    quantities instead:

    * ``dag/auroc_{block}``    -- ranking of attention against the true
      adjacency.  Invariant to any monotone rescaling; 0.5 = chance.  The
      honest answer to "is HSIC buying structure?".
    * ``dag/contrast_{block}`` -- mean attention on true edges minus mean on
      non-edges.  Directly interpretable, and NEGATIVE when the model attends
      to non-parents more than parents.
    * ``dag/mass_on_edges_{block}`` -- fraction of total attention mass sitting
      on true edges; falls when the posterior densifies.

    Enabled via ``training.log_dag_metrics: true``; cadence via
    ``training.dag_metrics_every_n_epochs`` (default 100).  Never interrupts
    training: every failure path degrades to a warning.

    Args:
        config:         Configuration dictionary (needs ``data.dataset``).
        data_dir:       Root data directory holding the true DAG masks.
        every_n_epochs: Evaluation cadence in epochs.
    """

    def __init__(self, config: dict, data_dir: str, every_n_epochs: int = 100):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.every_n_epochs = max(1, int(every_n_epochs))
        self._true_masks = None   # lazily loaded {block: ndarray | None}
        self._warned_no_att = False

    # -- helpers ---------------------------------------------------------
    def _load_true_masks(self):
        from causaliT.evaluation.eval_funs.helpers.eval_utils import _load_true_dag_mask
        dataset = self.config.get("data", {}).get("dataset")
        masks = {"cross": None, "self": None}
        if not dataset or not self.data_dir:
            return masks
        try:
            masks["cross"] = _load_true_dag_mask(self.data_dir, dataset, "dec_cross")
            masks["self"] = _load_true_dag_mask(self.data_dir, dataset, "dec_self")
        except Exception as e:   # pragma: no cover - diagnostics only
            logger.debug(f"PeriodicDAGMetrics: true-mask load failed: {e}")
        return masks

    @staticmethod
    def _auroc(scores: "np.ndarray", labels: "np.ndarray") -> float:
        """Rank-based AUROC via the Mann-Whitney U identity (ties averaged).

        Local implementation to avoid a hard sklearn dependency in the training
        loop; identical to ``roc_auc_score`` for binary labels.
        """
        import numpy as np
        pos = scores[labels > 0.5]
        neg = scores[labels <= 0.5]
        if pos.size == 0 or neg.size == 0:
            return float("nan")
        order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
        ranks = np.empty(order.size, dtype=float)
        ranks[order] = np.arange(1, order.size + 1, dtype=float)
        # Average ranks within tie groups so a constant matrix scores exactly .5
        vals = np.concatenate([pos, neg])[order]
        i = 0
        while i < vals.size:
            j = i
            while j + 1 < vals.size and vals[j + 1] == vals[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
            i = j + 1
        return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                     / (pos.size * neg.size))

    # -- lifecycle -------------------------------------------------------
    # Train-epoch end: ``_last_att_mean`` is stashed by ``_step`` on every
    # forward pass, so it is guaranteed fresh here.
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        import numpy as np

        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        att = getattr(pl_module, "_last_att_mean", None)
        if att is None:
            if not self._warned_no_att:
                self._warned_no_att = True
                logger.warning(
                    "PeriodicDAGMetrics: no attention posterior stashed on the "
                    "module (_last_att_mean is None) - dag/* metrics will be "
                    "absent from metrics.csv."
                )
            return

        if self._true_masks is None:
            self._true_masks = self._load_true_masks()

        # Split the posterior into the canonical S->X / X->X blocks using the
        # model's own shape-aware splitter (handles homogeneous (N, N) too).
        try:
            blocks = pl_module.split_attention_blocks(att.unsqueeze(0))
            learned = {
                "cross": blocks.get("s_to_x"),
                "self": blocks.get("x_to_x"),
            }
        except Exception as e:   # pragma: no cover - diagnostics only
            logger.warning(f"PeriodicDAGMetrics: attention split failed: {e}")
            return

        for block, mat in learned.items():
            true = self._true_masks.get(block)
            if mat is None or true is None:
                continue
            arr = mat.detach().cpu().numpy()
            arr = arr[0] if arr.ndim == 3 else arr
            if arr.shape != true.shape:
                logger.warning(
                    f"PeriodicDAGMetrics: shape mismatch [{block}] "
                    f"{arr.shape} vs true {true.shape}; skipping."
                )
                continue

            s, y = arr.ravel(), true.ravel()
            on, off = s[y > 0.5], s[y <= 0.5]
            if on.size == 0 or off.size == 0:
                continue

            pl_module.log(f"dag/auroc_{block}", self._auroc(s, y),
                          on_step=False, on_epoch=True)
            pl_module.log(f"dag/contrast_{block}",
                          float(on.mean() - off.mean()),
                          on_step=False, on_epoch=True)
            total = float(s.sum())
            if total > 1e-12:
                pl_module.log(f"dag/mass_on_edges_{block}",
                              float(on.sum() / total),
                              on_step=False, on_epoch=True)


class MetricsAggregator(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.logger.log_metrics(
            trainer.callback_metrics,
            step=trainer.current_epoch
        )
