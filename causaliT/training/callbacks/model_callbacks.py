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
    * ``dag/mass_on_parents_{block}`` -- same mass as ``mass_on_edges`` under
      the causal naming (true-edge cells ARE the direct-parent cells); kept as
      a separate column so the parents/ancestors/descendants/others partition
      is complete in every metrics.csv.
    * ``dag/mass_on_others_{block}`` -- fraction of mass on cells that are
      NEITHER parents/edges, ancestors, nor descendants (the unrelated
      residual).  parents + ancestors + descendants + others (+ diagonal, for
      square blocks) = 1.
    * ``dag/mass_on_ancestors_self`` -- fraction of the X->X attention mass on
      TRANSITIVE ancestors that are not direct parents (transitive closure of
      the true X->X mask minus the mask).  HSIC(res, X_j) is non-zero for
      every ancestor j unless the full parent set is conditioned on, so a
      rising ancestor share is the signature of the ancestor-shortcut regime;
      an effective L0 penalty should drive this share DOWN before it removes
      true-parent mass.
    * ``dag/parent_vs_ancestor_contrast_self`` -- mean attention on true-parent
      cells minus mean on (ancestor, not parent) cells.  Positive = the
      posterior prefers parents over their transitive proxies.

    Enabled via ``training.log_dag_metrics: true``; cadence via
    ``training.dag_metrics_every_n_epochs`` (default 50).  Metrics are always
    logged at epoch 0 and at the final epoch (also on early stopping via
    ``on_train_end``), so short runs still produce the full column set.
    Never interrupts training: every failure path degrades to a warning.

    Args:
        config:         Configuration dictionary (needs ``data.dataset``).
        data_dir:       Root data directory holding the true DAG masks.
        every_n_epochs: Evaluation cadence in epochs.
    """

    def __init__(self, config: dict, data_dir: str, every_n_epochs: int = 50):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.every_n_epochs = max(1, int(every_n_epochs))
        self._true_masks = None   # loaded in setup() {block: ndarray | None}
        self._ancestor_masks = {}  # lazily derived {block: ndarray | None}
        self._descendant_masks = {}  # lazily derived {block: ndarray | None}
        self._warned_no_att = False
        self._warned_skip = set()  # blocks already warned about (one-shot)
        self._last_logged_epoch = None  # for the on_train_end fallback

    @staticmethod
    def _ancestor_not_parent_mask(true: "np.ndarray") -> "np.ndarray":
        """Transitive-closure-minus-parents mask of a square DAG mask.

        ``true[i, j] = 1`` means ``j -> i``; the returned mask marks cells
        where j is an ANCESTOR of i but not a direct parent.  Boolean
        fixpoint iteration; all-zero (never wrong) if the input is not
        square or already equals its closure.
        """
        import numpy as np
        g = np.asarray(true) > 0.5
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            return np.zeros_like(g)
        tc = g.copy()
        for _ in range(g.shape[0]):
            new = tc | (tc @ g)
            if (new == tc).all():
                break
            tc = new
        return tc & ~g

    @staticmethod
    def _descendant_mask(true):
        # Mask of cells (i, j) where j is a DESCENDANT of i (anti-causal
        # mass).  true[i, j] = 1 means j -> i; j descendant of i iff i is an
        # ancestor of j, i.e. the TRANSPOSE of the ancestor relation.
        # Direct children are kept (a child IS a descendant); the diagonal
        # is absent by construction.  All-zero for non-square input.
        import numpy as np
        g = np.asarray(true) > 0.5
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            return np.zeros_like(g)
        tc = g.copy()
        for _ in range(g.shape[0]):
            new = tc | (tc @ g)
            if (new == tc).all():
                break
            tc = new
        return tc.T.copy()  # (i, j): i ancestor of j <=> j descendant of i

    # -- helpers ---------------------------------------------------------
    def _load_true_masks(self):
        from causaliT.evaluation.eval_funs.helpers.eval_utils import _load_true_dag_mask
        dataset = self.config.get("data", {}).get("dataset")
        masks = {"cross": None, "self": None}
        if not dataset or not self.data_dir:
            logger.warning(
                "PeriodicDAGMetrics: no data.dataset in config or no data_dir "
                "(dataset=%r, data_dir=%r) - dag/* metrics will be absent "
                "from metrics.csv.", dataset, self.data_dir,
            )
            return masks
        for block, mask_type in (("cross", "dec_cross"), ("self", "dec_self")):
            try:
                masks[block] = _load_true_dag_mask(self.data_dir, dataset, mask_type)
            except Exception as e:   # pragma: no cover - diagnostics only
                logger.warning(
                    f"PeriodicDAGMetrics: true-mask load failed [{block}] "
                    f"({mask_type}, dataset={dataset}): {e}"
                )
        return masks

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str = None):
        """Eagerly load the true DAG masks and report, once, which dag/* blocks
        will be (un)available -- so a missing column in metrics.csv always has
        a matching WARNING in the training log."""
        if stage is not None and stage != "fit":
            return
        if self._true_masks is None:
            self._true_masks = self._load_true_masks()
        for block, mask in self._true_masks.items():
            if mask is None:
                logger.warning(
                    f"PeriodicDAGMetrics: no true DAG mask for block "
                    f"'{block}' (dataset="
                    f"{self.config.get('data', {}).get('dataset')!r}) - "
                    f"dag/*_{block} columns will be absent from metrics.csv."
                )

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
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        max_epochs = getattr(trainer, "max_epochs", None)
        is_final = max_epochs is not None and epoch == max_epochs - 1
        if epoch % self.every_n_epochs != 0 and not is_final:
            return
        self._log_dag_metrics(trainer, pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        # Fallback for early stopping / cadence misses: guarantee one final
        # log so the dag/* columns always end the run with a value.
        if trainer.sanity_checking:
            return
        if self._last_logged_epoch == trainer.current_epoch:
            return
        try:
            self._log_dag_metrics(trainer, pl_module)
        except Exception as e:  # pragma: no cover - diagnostics only
            logger.warning(f"PeriodicDAGMetrics: final-epoch log failed: {e}")

    # Train-epoch end: ``_last_att_mean`` is stashed by ``_step`` on every
    # forward pass, so it is guaranteed fresh here.
    def _log_dag_metrics(self, trainer: Trainer, pl_module: LightningModule):
        import numpy as np

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
        self._last_logged_epoch = trainer.current_epoch

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
            anc_mass = 0.0  # filled below; used for the mass_on_others residual
            dsc_mass = 0.0
            if total > 1e-12:
                pl_module.log(f"dag/mass_on_edges_{block}",
                              float(on.sum() / total),
                              on_step=False, on_epoch=True)
                # Same quantity under the causal naming (true-edge cells ARE
                # the direct-parent cells) so the parents/ancestors/
                # descendants/others partition appears in every metrics.csv.
                pl_module.log(f"dag/mass_on_parents_{block}",
                              float(on.sum() / total),
                              on_step=False, on_epoch=True)

            # Ancestor-vs-parent split (square blocks only, i.e. X->X):
            # where the non-parent mass actually sits.  HSIC cannot reach its
            # zero floor through ancestors, so this tracks whether the
            # structural signal is being spent on transitive proxies.
            anc = self._ancestor_masks.get(block)
            if anc is None:
                anc = self._ancestor_not_parent_mask(true)
                self._ancestor_masks[block] = anc
            anc_flat = anc.ravel()
            if anc_flat.any():
                sa = s[anc_flat]
                anc_mass = float(sa.sum())
                if total > 1e-12:
                    pl_module.log(f"dag/mass_on_ancestors_{block}",
                                  float(anc_mass / total),
                                  on_step=False, on_epoch=True)
                pl_module.log(f"dag/parent_vs_ancestor_contrast_{block}",
                              float(on.mean() - sa.mean()),
                              on_step=False, on_epoch=True)

            # Descendant mass (square blocks only): the anti-causal channel -
            # fraction of mass on cells (i, j) with j a DESCENDANT of i
            # (reversed-direction edges).  Children are strong predictors of
            # their parents, so a rising share is the signature of the
            # posterior collapsing onto the anti-causal solution.
            dsc = self._descendant_masks.get(block)
            if dsc is None:
                dsc = self._descendant_mask(true)
                self._descendant_masks[block] = dsc
            dsc_flat = dsc.ravel()
            if dsc_flat.any():
                dsc_mass = float(s[dsc_flat].sum())
                if total > 1e-12:
                    pl_module.log(f"dag/mass_on_descendants_{block}",
                                  float(dsc_mass / total),
                                  on_step=False, on_epoch=True)

            # "Others": cells that are neither parents/edges, ancestors, nor
            # descendants -- the unrelated residual.  For non-square (cross)
            # blocks anc/dsc are all-zero and this equals the non-edge mass;
            # for square (self) blocks the diagonal is excluded as well.
            if total > 1e-12:
                others = total - float(on.sum()) - anc_mass - dsc_mass
                if arr.shape[0] == arr.shape[1]:
                    others -= float(np.diag(arr).sum())
                pl_module.log(f"dag/mass_on_others_{block}",
                              float(others / total),
                              on_step=False, on_epoch=True)


class HSICClassMetrics(PeriodicDAGMetrics):
    """Log PRE-WEIGHTING HSIC nodal contributions split by causal class.

    Reads the raw pair HSIC matrix stashed by the forecaster
    (``_last_hsic_pair_mat``, detached, NaN = excluded pair) and reports,
    per block, the NaN-safe MEAN of the raw HSIC over each class of the
    true-DAG partition -- the same parents/ancestors/descendants/others
    partition as ``dag/mass_on_*``, but for the EVIDENCE (the H values the
    weighting rules tilt by), not the attention mass:

    * ``hsic_class/parents_self``      -- should fall to the independence
      floor as parents get fitted (coverage signal under a growing top-k);
    * ``hsic_class/descendants_self``  -- must stay strictly ABOVE the floor
      (irreducible under an ANM): the quantitative "reintroducing a
      descendant is expensive" check;
    * ``hsic_class/ancestors_self`` / ``hsic_class/others_self`` -- the
      transitive-proxy and unrelated floors;
    * ``hsic_class/parents_cross`` / ``hsic_class/others_cross`` -- the S->X
      block has no transitive structure, parents vs others only;
    * ``hsic_class/n_*``               -- cell counts, logged once.

    Block extraction mirrors ``split_attention_blocks``: split mode
    ``(n_X, L_S + L_X)`` cuts columns at ``L_S``; homogeneous mode
    ``(N, N)`` takes X-child rows first (``[L_S:]``), then S/X columns.

    Same gating/cadence as ``PeriodicDAGMetrics`` (``training.log_dag_metrics``,
    ``dag_metrics_every_n_epochs``), epoch 0 + final-epoch guarantee, and it
    never interrupts training: a missing matrix (non-softmax aggregation)
    warns once and produces no columns.
    """

    def __init__(self, config: dict, data_dir: str, every_n_epochs: int = 50):
        super().__init__(config, data_dir, every_n_epochs)
        self._warned_no_mat = False
        self._counts_logged = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        max_epochs = getattr(trainer, "max_epochs", None)
        is_final = max_epochs is not None and epoch == max_epochs - 1
        if epoch % self.every_n_epochs != 0 and not is_final:
            return
        self._log_hsic_class_metrics(trainer, pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.sanity_checking:
            return
        if self._last_logged_epoch == trainer.current_epoch:
            return
        try:
            self._log_hsic_class_metrics(trainer, pl_module)
        except Exception as e:  # pragma: no cover - diagnostics only
            logger.warning(f"HSICClassMetrics: final-epoch log failed: {e}")

    def _log_hsic_class_metrics(self, trainer: Trainer, pl_module: LightningModule):
        import numpy as np

        H = getattr(pl_module, "_last_hsic_pair_mat", None)
        if H is None:
            if not self._warned_no_mat:
                self._warned_no_mat = True
                logger.warning(
                    "HSICClassMetrics: no pair HSIC matrix stashed on the "
                    "module (_last_hsic_pair_mat is None) - hsic_class/* "
                    "metrics need an attw_softmax aggregation branch; "
                    "columns will be absent from metrics.csv."
                )
            return
        self._last_logged_epoch = trainer.current_epoch

        if self._true_masks is None:
            self._true_masks = self._load_true_masks()

        arr = H.detach().cpu().numpy().astype(float)
        arr = arr[0] if arr.ndim == 3 else arr
        homogeneous = bool(getattr(pl_module, "homogeneous_nodes", False))
        L_S = getattr(pl_module, "S_seq_len", None)
        if L_S is None:
            if not self._warned_no_mat:
                self._warned_no_mat = True
                logger.warning(
                    "HSICClassMetrics: module has no S_seq_len; cannot split "
                    "the pair matrix into blocks - hsic_class/* absent."
                )
            return
        if homogeneous:
            # (N, N): X-child rows first, then S / X parent columns.
            blocks = {"cross": arr[L_S:, :L_S], "self": arr[L_S:, L_S:]}
        else:
            # (n_X, L_S + L_X): cut columns at L_S.
            blocks = {"cross": arr[:, :L_S], "self": arr[:, L_S:]}

        for block, mat in blocks.items():
            true = self._true_masks.get(block)
            if true is None:
                continue   # setup() already warned about the missing mask
            if mat.shape != true.shape:
                if block not in self._warned_skip:
                    self._warned_skip.add(block)
                    logger.warning(
                        f"HSICClassMetrics: shape mismatch [{block}] "
                        f"{mat.shape} vs true {true.shape}; skipping."
                    )
                continue
            finite = np.isfinite(mat)
            parents = true > 0.5
            if mat.shape[0] == mat.shape[1]:
                anc = self._ancestor_not_parent_mask(true)
                dsc = self._descendant_mask(true)
                others = ~(parents | anc | dsc) & ~np.eye(
                    mat.shape[0], dtype=bool
                )
                classes = {"parents": parents, "ancestors": anc,
                           "descendants": dsc, "others": others}
            else:
                classes = {"parents": parents, "others": ~parents}
            for cls, m in classes.items():
                sel = m & finite
                vals = mat[sel]
                if vals.size == 0:
                    continue
                pl_module.log(f"hsic_class/{cls}_{block}",
                              float(vals.mean()),
                              on_step=False, on_epoch=True)
                if not self._counts_logged:
                    pl_module.log(f"hsic_class/n_{cls}_{block}",
                                  float(int(sel.sum())),
                                  on_step=False, on_epoch=True)
        self._counts_logged = True


class MetricsAggregator(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.logger.log_metrics(
            trainer.callback_metrics,
            step=trainer.current_epoch
        )
