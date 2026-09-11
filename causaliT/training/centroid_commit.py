"""
Centroid-commit query dynamics ("quantized queries with evidence shadow").

Idea
----
The structural query of each node is not free-continuous: it COMMITS to the
centroid of a subset of the (frozen, orthonormal) key frame - i.e. to a
discrete parent-set hypothesis.  Between commits the forward query is frozen,
so per-step gradient noise cannot diffuse the posterior (the measured failure
mode of the continuous scheme).  Evidence accumulates in a per-node SHADOW:

    shadow_i <- committed_i + beta * (shadow_i - committed_i) - eta * g_i

where g_i is the (HSIC-only by default) structural gradient row.  ``shadow_i
- committed_i`` is the leaked accumulated evidence.  When the exact
nearest-centroid projection of the shadow selects a DIFFERENT subset than the
committed one, the node commits: weight <- c(S'), shadow <- c(S'), M_i reset.

Exact projection (orthonormal keys): with a_j = q_hat . k_j,
cos(q, c_S) = sum_{j in S} a_j / sqrt(|S|), so the best subset is the top-m
keys by a_j for the best m - O(N log N), no 2^N enumeration.  m = 0 (empty
subset, zero query) is included: it is the right hypothesis for sources.

On commit M_i (the per-node norm budget) is optionally reset so the new
hypothesis opens its gates (``reset_m_on_commit='one'``; ``'off'`` leaves M to
the optimizer - resetting inflates the edge logits of the whole committed
subset and was observed to kill the post-commit gradient signal).

Additional commit-rule options:

* ``prior_rho``: per-key prior penalty in the subset score
  (``sum(a)/sqrt(m) - rho*m``) - MAP instead of ML, counteracts the dense-set
  bias of diffuse alignments at the full-centroid init.
* ``winner_take_all``: at most one commit per step, the eligible node with
  the largest score gain (coordinate-wise MAP conditioned on the committed
  state of the other nodes).
* ``commit_margin``: hysteresis - minimum prior-adjusted score gain required.
* ``min_snr``: noise floor on the per-node gradient SNR (streaming EMA, as in
  NodewiseQuerySelector); below it no commit happens for that node.

Wire-in: the forecaster builds a CentroidCommitController when
``training.centroid_commit.enabled``; per structural step it calls
``controller.step()`` after the structural backward (shadow grads come from
the STE path in FreeQueryEmbedding.forward).
"""

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def subset_score(a_sorted_csum: torch.Tensor, m: int,
                 prior_rho: float = 0.0) -> float:
    """Prior-adjusted score of the top-m subset: sum(a)/sqrt(m) - rho * m.

    ``cos(q, c_S) = sum_{j in S} a_j / sqrt(|S|)`` is the profile
    log-likelihood of S under a vMF noise model on the sphere; ``prior_rho``
    is a per-key prior penalty (log-odds units) turning pure ML - which favours
    dense sets under diffuse alignments - into MAP.
    """
    if m < 1:
        return 0.0
    return float(a_sorted_csum[m - 1]) / float(m) ** 0.5 - prior_rho * m


def best_subset(q: torch.Tensor, K: torch.Tensor,
                exclude: Optional[int] = None,
                prior_rho: float = 0.0) -> torch.Tensor:
    """Exact argmax_S [cos(q, centroid(K[S])) - prior_rho * |S|].

    Args:
        q:         (d,) query (need not be normalised).
        K:         (n, d) orthonormal key frame.
        exclude:   optional key index forbidden from the subset (self-loop).
        prior_rho: per-key prior penalty on the subset size (0 = pure ML, the
                   original behaviour).

    Returns:
        Long tensor of selected key indices (alignment-desc order; compare
        subsets canonically with ``torch.equal(a.sort().values, ...)``).
    """
    q = q.detach()
    K = K.to(q.device)
    a = K @ (q / q.norm().clamp_min(1e-12))          # (n,) alignments
    if exclude is not None:
        a = a.clone()
        a[exclude] = -float("inf")
    a_sorted, order = torch.sort(a, descending=True)
    csum = torch.cumsum(a_sorted, dim=0)              # sum of top-m
    scores = torch.tensor(
        [subset_score(csum, m, prior_rho) for m in range(1, len(a) + 1)],
        dtype=a.dtype,
    )
    best = int(scores.argmax())
    if float(scores[best]) <= 0.0:
        return torch.empty(0, dtype=torch.long, device=a.device)  # empty subset
    return order[: best + 1]


def best_subset_excluding(q: torch.Tensor, K: torch.Tensor,
                          exclude: Optional[int] = None,
                          prior_rho: float = 0.0,
                          forbidden: Optional[set] = None) -> torch.Tensor:
    """``best_subset`` with forbidden subsets (soft taboo for rejected commits).

    Exact: the optimal subset of size m is always the top-m keys by
    alignment, so enumerating m = 0..n covers every candidate; forbidden
    subsets (frozensets of key indices, e.g. bilevel-rejected centroids) are
    skipped and the next-best admissible subset is returned.  The empty
    subset is returned when every admissible subset scores <= 0.
    """
    forbidden = forbidden or set()
    q = q.detach()
    K = K.to(q.device)
    a = K @ (q / q.norm().clamp_min(1e-12))
    if exclude is not None:
        a = a.clone()
        a[exclude] = -float("inf")
    a_sorted, order = torch.sort(a, descending=True)
    csum = torch.cumsum(a_sorted, dim=0)
    best_subset_t = torch.empty(0, dtype=torch.long, device=a.device)
    best_score = 0.0                     # empty subset has score 0
    for m in range(1, len(a) + 1):
        cand = order[:m]
        if frozenset(cand.tolist()) in forbidden:
            continue
        s = subset_score(csum, m, prior_rho)
        if s > best_score:
            best_score = s
            best_subset_t = cand
    return best_subset_t


def subset_score_of(a: torch.Tensor, S: torch.Tensor,
                    prior_rho: float = 0.0) -> float:
    """Prior-adjusted score of an explicit subset S given alignments a."""
    if S.numel() == 0:
        return 0.0
    return float(a[S].sum()) / float(S.numel()) ** 0.5 - prior_rho * S.numel()


def centroid_of(K: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    """Normalised centroid of K[subset]; the zero vector for the empty set."""
    if subset.numel() == 0:
        return torch.zeros(K.shape[1], dtype=K.dtype, device=K.device)
    c = K[subset].mean(dim=0)
    return c / c.norm().clamp_min(1e-12)



class CentroidCommitController:
    """Per-node evidence accumulation + centroid commits.

    Parameters
    ----------
    tables :
        The FreeQueryEmbedding tables in node order (e.g. ``[query_embed_S,
        query_embed_X]``); row 0 of each is padding.  Node i maps to the
        concatenated real rows.
    K :
        (n_nodes, d) frozen orthonormal key frame, in the same node order, OR
        a zero-argument callable returning it (evaluated at every step, so a
        frame loaded from a checkpoint after construction is picked up).
    norm_param :
        ``query_norm_log_scale`` (n_nodes,), tied across blocks; row i is
        reset on commits of node i when ``reset_m_on_commit`` is set.
    """

    def __init__(
        self,
        tables: list,
        K: torch.Tensor,
        norm_param: Optional[torch.Tensor],
        evidence_lr: float = 1.0,
        evidence_leak: float = 0.95,
        reset_m_on_commit: str = "one",
        prior_rho: float = 0.0,
        winner_take_all: bool = False,
        commit_margin: float = 0.0,
        min_snr: float = 0.0,
    ):
        if reset_m_on_commit not in ("one", "off"):
            raise ValueError(
                f"reset_m_on_commit must be 'one' or 'off', got {reset_m_on_commit!r}"
            )
        self.tables = tables
        self._key_fn = K if callable(K) else None
        self.K = (K() if callable(K) else K).detach().double().cpu()
        self.norm_param = norm_param
        self.eta = float(evidence_lr)
        self.beta = float(evidence_leak)
        self.reset_m_on_commit = reset_m_on_commit
        self.prior_rho = float(prior_rho)
        self.winner_take_all = bool(winner_take_all)
        self.commit_margin = float(commit_margin)
        self.min_snr = float(min_snr)

        # Node map: node i -> (table, row, global key index of the node
        # itself (forbidden from its own subset: no self-loops).
        self.node_map: List = []
        gi = 0
        for t in tables:
            for r in range(1, t.embedding.weight.shape[0]):
                self.node_map.append((t, r, gi))
                gi += 1
        self.n_nodes = gi
        assert self.K.shape[0] == self.n_nodes, (
            f"key frame has {self.K.shape[0]} rows but the tables expose "
            f"{self.n_nodes} nodes."
        )
        # SNR evidence (mirrors NodewiseQuerySelector, Option E): EMA of the
        # per-node gradient mean and second moment, bias-corrected, reset for a
        # node on commit (post-commit evidence must condition on the new
        # committed state).  Only maintained when ``min_snr > 0``.
        # Kept on CPU permanently: the controller is built before Lightning
        # moves the model to its device, and the EMA is tiny (n_nodes x d), so
        # the per-step transfer of the gradient row is the cheap direction.
        d = self.tables[0].embedding.weight.shape[1]
        self._snr_mean = torch.zeros(self.n_nodes, d, dtype=torch.float64)
        self._snr_sq = torch.zeros(self.n_nodes, dtype=torch.float64)
        self._snr_t = torch.zeros(self.n_nodes, dtype=torch.long)
        self._d = d

        # Diagnostics.
        self.n_commits = 0
        self.commit_counts = torch.zeros(self.n_nodes, dtype=torch.long)
        self.last_commits: List[Dict] = []   # per-step: node, margin, sizes

        # Bilevel gate state (docs/ideas/BILEVEL_CENTROID_COMMIT.md).
        # ``taboos[i]``: frozensets of key indices whose commit was REJECTED
        # by the paired refit probe; while tabooed the node's candidacy uses
        # the best NON-taboo subset.  A taboo lifts automatically when the
        # shadow's raw projection is neither the committed subset nor the
        # tabooed one (the shadow moved away; re-entry stays possible).
        # ``pending``: eligible commits awaiting the probe (defer mode).
        self.taboos: List[set] = [set() for _ in range(self.n_nodes)]
        self.pending: List[Dict] = []
        self.last_rejects: List[Dict] = []

    # ------------------------------------------------------------------
    def _update_snr(self, i: int, g: torch.Tensor) -> None:
        """Streaming EMA of the per-node gradient mean / second moment."""
        g = g.double().cpu()
        self._snr_mean[i].mul_(self.beta).add_(g, alpha=1.0 - self.beta)
        self._snr_sq[i].mul_(self.beta).add_(
            float(g.pow(2).sum()), alpha=1.0 - self.beta)
        self._snr_t[i] += 1

    def current_snr(self) -> torch.Tensor:
        """Bias-corrected per-node SNR = ||mu||^2 / (var trace / d)."""
        bc = 1.0 - float(self.beta) ** self._snr_t.clamp(min=1).double()
        bc = bc.to(self._snr_mean.device)
        mu = self._snr_mean / bc[:, None]
        sq = self._snr_sq / bc
        var = (sq - mu.pow(2).sum(dim=1)).clamp_min(0.0) / self._d
        return mu.pow(2).sum(dim=1) / (var + 1e-12)

    def _commit(self, i: int, t, r: int, gi: int, cand: torch.Tensor,
                cur_size: int, margin: float) -> None:
        """Write the new centroid, re-centre the shadow, reset the evidence."""
        w = t.embedding.weight
        c = centroid_of(self.K, cand).to(dtype=w.dtype, device=w.device)
        w[r].copy_(c)
        t.shadow[r].copy_(c)
        if self.reset_m_on_commit == "one" and self.norm_param is not None:
            self.norm_param.data[gi] = 0.0   # M_i = exp(0) = 1
        self._snr_mean[i].zero_()
        self._snr_sq[i].zero_()
        self._snr_t[i] = 0
        self.n_commits += 1
        self.commit_counts[i] += 1
        self.last_commits.append({"node": gi, "margin": margin,
                                  "from": cur_size, "to": int(cand.numel())})
        logger.info(
            "centroid commit: node %d |S|: %d -> %d (margin %.4f)",
            gi, cur_size, cand.numel(), margin,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, grad_override: Optional[list] = None,
             defer: bool = False) -> int:
        """One evidence-accumulation step + commit check.  Returns #eligible.

        Consumes the shadow gradients and clears them.  ``grad_override``
        (list aligned to ``tables``, one (rows, d) tensor or None per table)
        replaces ``shadow.grad`` - used to feed HSIC-only evidence instead of
        the total structural gradient.  No-op for rows without grads.

        Commit eligibility per node: the shadow's prior-adjusted best subset
        differs from the committed one, the score gain is at least
        ``commit_margin`` (hysteresis), and - when ``min_snr > 0`` - the
        node's gradient SNR clears the noise floor.  With
        ``winner_take_all`` only the eligible node with the largest margin
        commits (a coordinate-wise MAP update conditioned on the committed
        state of the other nodes); otherwise all eligible nodes commit.

        Bilevel gate (``defer=True``): eligible commits are NOT applied; they
        are stored in ``self.pending`` for the caller to probe (see
        ``causaliT.training.bilevel_probe``) and then finalised via
        :meth:`finalize`.  Tabooed (previously rejected) subsets are skipped
        during candidacy; a taboo lifts once the shadow's raw projection is
        neither the committed nor the tabooed subset.
        """
        if self._key_fn is not None:
            self.K = self._key_fn().detach().double().cpu()
        self.last_commits = []
        self.last_rejects = []
        self.pending = []

        # Pass 1: evidence update for every node + candidacy evaluation.
        eligible: List[Dict] = []
        for i, (t, r, gi) in enumerate(self.node_map):
            if grad_override is not None:
                ti = self.tables.index(t)
                gtab = grad_override[ti]
                g = gtab[r] if gtab is not None else None
            else:
                g = t.shadow.grad[r] if t.shadow.grad is not None else None
            if g is None:
                continue
            self._update_snr(i, g)
            w = t.embedding.weight
            evid = self.beta * (t.shadow[r] - w[r]).double() - self.eta * g.double()
            t.shadow[r] = w[r] + evid.to(t.shadow.dtype)
            sh = t.shadow[r].detach().double().cpu()
            cur = best_subset(w[r].detach().double().cpu(), self.K, exclude=gi,
                              prior_rho=self.prior_rho)
            raw = best_subset(sh, self.K, exclude=gi, prior_rho=self.prior_rho)

            # Taboo maintenance: a tabooed subset stays forbidden while the
            # shadow's RAW projection is the committed subset or the tabooed
            # one; it lifts as soon as the shadow points elsewhere.
            if self.taboos[i]:
                raw_fs, cur_fs = frozenset(raw.tolist()), frozenset(cur.tolist())
                self.taboos[i] = {T for T in self.taboos[i]
                                  if raw_fs in (cur_fs, T)}

            # Candidacy: if the raw projection is tabooed, fall back to the
            # best admissible (non-taboo) subset.
            if self.taboos[i] and frozenset(raw.tolist()) in self.taboos[i]:
                cand = best_subset_excluding(sh, self.K, exclude=gi,
                                             prior_rho=self.prior_rho,
                                             forbidden=self.taboos[i])
            else:
                cand = raw
            same = (cand.numel() == cur.numel()
                    and torch.equal(cand.sort().values, cur.sort().values))
            if same:
                continue
            # Score gain of the candidate over the committed subset, evaluated
            # on the SHADOW alignments (the evidence's own geometry).
            # ``sh`` is CPU/double by construction; ``self.K`` is too when it
            # comes from ``_key_fn``, but a caller-supplied frame may live on
            # CUDA — align devices so the margin never raises a mismatch.
            a = self.K.to(sh.device) @ (sh / sh.norm().clamp_min(1e-12))
            a[gi] = -float("inf")
            margin = (subset_score_of(a, cand, self.prior_rho)
                      - subset_score_of(a, cur, self.prior_rho))
            if margin < self.commit_margin:
                continue
            eligible.append({"i": i, "t": t, "r": r, "gi": gi, "cand": cand,
                             "cur_size": int(cur.numel()), "margin": margin})

        # SNR noise floor (evaluated AFTER this step's EMA update).
        if self.min_snr > 0 and eligible:
            snr = self.current_snr()
            eligible = [e for e in eligible
                        if float(snr[e["i"]]) >= self.min_snr]

        # Pass 2: commit (or defer to the bilevel gate).
        if self.winner_take_all and len(eligible) > 1:
            eligible = [max(eligible, key=lambda e: e["margin"])]
        if defer:
            self.pending = eligible
        else:
            for e in eligible:
                self._commit(e["i"], e["t"], e["r"], e["gi"], e["cand"],
                             e["cur_size"], e["margin"])

        for t, _, _ in self.node_map:
            if t.shadow.grad is not None:
                t.shadow.grad = None
        return len(eligible)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def finalize(self, entry: Dict, accepted: bool) -> None:
        """Resolve one deferred (probed) candidacy from ``step(defer=True)``.

        accepted=True  -> commit exactly as the ungated path would.
        accepted=False -> the candidate subset is tabooed; the shadow is NOT
        reset (evidence keeps accumulating, so the node can diffuse out of
        the tabooed centroid and re-enter it later once the taboo lifts).
        """
        if accepted:
            self._commit(entry["i"], entry["t"], entry["r"], entry["gi"],
                         entry["cand"], entry["cur_size"], entry["margin"])
        else:
            self.taboos[entry["i"]].add(frozenset(entry["cand"].tolist()))
            self.last_rejects.append({"node": entry["gi"],
                                      "size": int(entry["cand"].numel())})
            logger.info(
                "centroid commit REJECTED by bilevel gate: node %d |S'|: %d "
                "(tabooed; %d taboos on this node)",
                entry["gi"], int(entry["cand"].numel()),
                len(self.taboos[entry["i"]]),
            )

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        """Persistent controller state (taboos + counters) for checkpoints."""
        return {
            "taboos": [[sorted(T) for T in s] for s in self.taboos],
            "n_commits": self.n_commits,
            "commit_counts": self.commit_counts.clone(),
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.taboos = [{frozenset(T) for T in s} for s in sd["taboos"]]
        self.n_commits = int(sd["n_commits"])
        self.commit_counts = sd["commit_counts"].clone()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def assignment_sizes(self) -> torch.Tensor:
        """Current |S_i| per node (n_nodes,)."""
        if self._key_fn is not None:
            self.K = self._key_fn().detach().double().cpu()
        out = torch.zeros(self.n_nodes, dtype=torch.long)
        for i, (t, r, gi) in enumerate(self.node_map):
            out[i] = best_subset(
                t.embedding.weight[r].detach().double().cpu(), self.K,
                exclude=gi, prior_rho=self.prior_rho,
            ).numel()
        return out
