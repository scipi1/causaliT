"""
Random SCM / DAG sampling for `scm_ds`.

This module adds the ability to *sample* random Structural Causal Models (and their
corresponding datasets) from a compact configuration, instead of authoring every SCM
explicitly in `datasets.py`. It is designed to stress-test models on larger DAGs and
across many structural regimes (size x degree x linearity x noise) in a reproducible
way.

Design
------
The neat integration point is the existing `SCMDataset`: if we can emit valid
`specs` (NodeSpec list), `params`, `singles` (noise samplers) and the
`source/input/target` label lists, we inherit the entire downstream pipeline for free
(adjacency, attention masks, metadata, ATE ground-truth, normalization, splits, graph
visualization). Hence this module only *builds ingredients* and returns an
`SCMDataset`; `scm.py` is left untouched.

Key properties
--------------
- Reproducibility: a single `seed` drives DAG sampling, structural-equation choices,
  weight sampling and per-node noise-type choices. Same config + seed => identical SCM.
- ER-k benchmark fidelity: `degree` = expected number of edges per node, so the target
  total edge count is `m = round(degree * n_nodes)` (ER1 -> n edges, ER2 -> 2n, ...).
  Edges are sampled *exactly* (uniformly, without replacement) by default.
- S/X paradigm: the first `n_sources` nodes are sources `S` (no incoming edges); the
  rest are inputs `X`. Only `S->X` and `X->X` edges are allowed, matching the
  cross-attention / self-attention mask structure used by the model. `target_labels`
  is empty by design (no `Y` decoder stage for now).
- Variance control: with `rescale_by_indegree=True` each parent weight is baked as
  `w / sqrt(in_degree)`, keeping per-node signal variance roughly stable as DAGs grow.

The full `RandomSCMConfig` is stored in the dataset's `meta` so any generated dataset
is exactly regenerable from its `meta.json`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from scm_ds.scm import NodeSpec, SCMDataset


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RandomSCMConfig:
    """
    Configuration describing a random SCM/DAG to sample.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes in the DAG (S sources + X inputs).
    degree : float
        ER-k expected number of edges *per node*. Target total edges is
        ``m = round(degree * n_nodes)``.
    seed : int
        Master seed. Same config + seed always yields the same SCM/DAG.
    linearity : {"linear", "nonlinear", "mixed"}
        Structural-equation family. "mixed" chooses per-node.
    noise : {"gaussian", "nongaussian", "mixed"}
        Noise family for the X (input) nodes. "mixed" chooses per-node.
    n_sources : Optional[int]
        Explicit number of source (S) nodes. Takes precedence over ``s_x_ratio``.
    s_x_ratio : Optional[float]
        Fraction of nodes that are sources, in (0, 1). Used only if ``n_sources``
        is None. If both are None, defaults to 0.3.
    weight_range : Tuple[float, float]
        Range for the *magnitude* of structural weights; sign is random.
    noise_scale : float
        Scale applied to X-node noise (sources use bounded ``source_noise``).
    source_noise : {"uniform", "gaussian"}
        Exogenous noise for S sources. "uniform" gives bounded support so that
        do(S=.) interventions stay in-distribution (matches existing datasets).
    nonlinear_fns : Tuple[str, ...]
        Pool of nonlinear link functions to draw from. Supported:
        "square", "cube", "sin", "tanh".
    rescale_by_indegree : bool
        If True, bake ``w / sqrt(in_degree)`` into weights (variance control).
    ensure_x_has_parent : bool
        If True, guarantee every X node has at least one parent (keeps S/X counts
        exact and prevents X nodes from silently becoming roots).
    exact_edges : bool
        If True, sample exactly ``m`` edges (uniform, no replacement). If False,
        use Bernoulli(p) with ``p = m / n_slots`` (random total edge count).
    name : Optional[str]
        Optional dataset name; a descriptive default is generated if omitted.
    """
    n_nodes: int
    degree: float
    seed: int
    linearity: str = "linear"
    noise: str = "gaussian"
    n_sources: Optional[int] = None
    s_x_ratio: Optional[float] = None
    weight_range: Tuple[float, float] = (0.5, 2.0)
    noise_scale: float = 0.1
    source_noise: str = "uniform"
    nonlinear_fns: Tuple[str, ...] = ("square", "cube", "sin", "tanh")
    rescale_by_indegree: bool = True
    ensure_x_has_parent: bool = True
    exact_edges: bool = True
    name: Optional[str] = None

    def resolved_n_sources(self) -> int:
        """Resolve the number of source nodes from n_sources / s_x_ratio."""
        if self.n_sources is not None:
            n_s = int(self.n_sources)
        else:
            ratio = self.s_x_ratio if self.s_x_ratio is not None else 0.3
            n_s = max(1, round(ratio * self.n_nodes))
        return n_s


# --------------------------------------------------------------------------- #
# DAG sampling
# --------------------------------------------------------------------------- #

def _sample_dag(cfg: RandomSCMConfig, rng: np.random.Generator) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    """
    Sample a reproducible DAG respecting the S/X structure.

    Nodes are laid out in a fixed topological order: the first ``n_sources`` positions
    are S sources (no incoming edges), the remaining positions are X inputs. Allowed
    edges go from an earlier position to a later position and always point *into* an X
    node (S->X or X->X). This guarantees acyclicity by construction.

    Returns
    -------
    names : list of node names in topological order
    source_labels : list of S node names
    input_labels : list of X node names
    parents : dict mapping each node name -> ordered list of parent names
    """
    n = int(cfg.n_nodes)
    if n < 2:
        raise ValueError("n_nodes must be >= 2 (need at least one source and one input).")

    n_sources = cfg.resolved_n_sources()
    if not (1 <= n_sources <= n - 1):
        raise ValueError(
            f"n_sources={n_sources} invalid for n_nodes={n}; must be in [1, n_nodes-1]."
        )

    # Node names by topological position: S first, then X.
    def _name(i: int) -> str:
        if i < n_sources:
            return f"S{i + 1}"
        return f"X{i - n_sources + 1}"

    names = [_name(i) for i in range(n)]
    source_labels = names[:n_sources]
    input_labels = names[n_sources:]

    target_m = round(cfg.degree * n)

    # All allowed (parent_pos, child_pos) slots: child must be an X (pos >= n_sources),
    # parent must be strictly earlier.
    all_slots: List[Tuple[int, int]] = [
        (p, c) for c in range(n_sources, n) for p in range(c)
    ]

    edges: set[Tuple[int, int]] = set()

    # Guarantee each X node has at least one parent.
    if cfg.ensure_x_has_parent:
        for c in range(n_sources, n):
            p = int(rng.integers(0, c))  # 0..c-1
            edges.add((p, c))

    remaining = [e for e in all_slots if e not in edges]
    n_extra = target_m - len(edges)

    if cfg.exact_edges:
        if n_extra > 0 and remaining:
            k = min(n_extra, len(remaining))
            idx = rng.choice(len(remaining), size=k, replace=False)
            for i in np.atleast_1d(idx):
                edges.add(remaining[int(i)])
    else:
        # Bernoulli(p) over the remaining slots to reach ~target_m in expectation.
        denom = len(all_slots) if all_slots else 1
        p = min(1.0, max(0.0, target_m / denom))
        for e in remaining:
            if rng.random() < p:
                edges.add(e)

    # Build ordered parent lists (parents sorted by topological position for stability).
    parents: Dict[str, List[str]] = {nm: [] for nm in names}
    for (p, c) in sorted(edges):
        parents[names[c]].append(names[p])

    return names, source_labels, input_labels, parents


# --------------------------------------------------------------------------- #
# Structural-equation builder
# --------------------------------------------------------------------------- #

_NONLINEAR_TEMPLATES = {
    "square": "{w}*{p}**2",
    "cube": "{w}*{p}**3",
    "sin": "{w}*sin({p})",
    "tanh": "{w}*tanh({p})",
    "linear": "{w}*{p}",
}


def _sample_weight(cfg: RandomSCMConfig, rng: np.random.Generator, in_degree: int) -> float:
    """Sample a signed weight; optionally rescale by sqrt(in_degree)."""
    lo, hi = cfg.weight_range
    mag = rng.uniform(lo, hi)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    w = sign * mag
    if cfg.rescale_by_indegree and in_degree > 0:
        w = w / math.sqrt(in_degree)
    return float(w)


def _build_specs_and_params(
    cfg: RandomSCMConfig,
    names: List[str],
    source_labels: List[str],
    parents: Dict[str, List[str]],
    rng: np.random.Generator,
) -> Tuple[List[NodeSpec], Dict[str, float]]:
    """
    Build NodeSpec list and numeric params for the sampled DAG.

    Sources are exogenous: ``expr = eps_<S>``.
    Each X node combines its parents (linear / nonlinear / mixed) plus its own noise.
    """
    source_set = set(source_labels)
    specs: List[NodeSpec] = []
    params: Dict[str, float] = {}

    for node in names:
        node_parents = parents[node]

        if node in source_set or len(node_parents) == 0:
            # Exogenous source (or a parentless node): pure noise.
            specs.append(NodeSpec(node, list(node_parents), f"eps_{node}"))
            continue

        in_degree = len(node_parents)

        # Decide the family for this node.
        if cfg.linearity == "linear":
            node_nonlinear = False
        elif cfg.linearity == "nonlinear":
            node_nonlinear = True
        elif cfg.linearity == "mixed":
            node_nonlinear = bool(rng.random() < 0.5)
        else:
            raise ValueError(f"Unknown linearity: {cfg.linearity!r}")

        terms: List[str] = []
        for parent in node_parents:
            w_key = f"w_{node}_{parent}"
            params[w_key] = _sample_weight(cfg, rng, in_degree)

            if node_nonlinear:
                fn = str(rng.choice(list(cfg.nonlinear_fns)))
            else:
                fn = "linear"
            template = _NONLINEAR_TEMPLATES[fn]
            terms.append(template.format(w=w_key, p=parent))

        expr = " + ".join(terms) + f" + eps_{node}"
        specs.append(NodeSpec(node, list(node_parents), expr))

    return specs, params


# --------------------------------------------------------------------------- #
# Noise builders
# --------------------------------------------------------------------------- #

def _gaussian_sampler(scale: float):
    return lambda rng, n: scale * rng.standard_normal(n)


def _uniform_source_sampler():
    return lambda rng, n: rng.uniform(-1.0, 1.0, n)


def _gaussian_source_sampler():
    return lambda rng, n: rng.standard_normal(n)


def _uniform_sampler(scale: float):
    return lambda rng, n: scale * rng.uniform(-1.0, 1.0, n)


def _exponential_sampler(scale: float):
    return lambda rng, n: scale * (rng.exponential(1.0, n) - 1.0)


def _laplace_sampler(scale: float):
    return lambda rng, n: scale * rng.laplace(0.0, 1.0, n)


def _lognormal_sampler(scale: float):
    return lambda rng, n: scale * (rng.lognormal(0.0, 0.5, n) - 1.0)


_NONGAUSSIAN_FACTORIES = {
    "uniform": _uniform_sampler,
    "exponential": _exponential_sampler,
    "laplace": _laplace_sampler,
    "lognormal": _lognormal_sampler,
}


def _build_singles(
    cfg: RandomSCMConfig,
    source_labels: List[str],
    input_labels: List[str],
    rng: np.random.Generator,
) -> Dict[str, object]:
    """
    Build per-node noise samplers.

    Sources use the bounded (or gaussian) ``source_noise``. X nodes use the requested
    noise family; for "nongaussian"/"mixed" the concrete distribution is chosen per
    node reproducibly from ``rng``.
    """
    singles: Dict[str, object] = {}

    # Source noise
    if cfg.source_noise == "uniform":
        for s in source_labels:
            singles[s] = _uniform_source_sampler()
    elif cfg.source_noise == "gaussian":
        for s in source_labels:
            singles[s] = _gaussian_source_sampler()
    else:
        raise ValueError(f"Unknown source_noise: {cfg.source_noise!r}")

    # Input (X) noise
    nongaussian_names = list(_NONGAUSSIAN_FACTORIES.keys())
    for x in input_labels:
        if cfg.noise == "gaussian":
            use_gaussian = True
        elif cfg.noise == "nongaussian":
            use_gaussian = False
        elif cfg.noise == "mixed":
            use_gaussian = bool(rng.random() < 0.5)
        else:
            raise ValueError(f"Unknown noise: {cfg.noise!r}")

        if use_gaussian:
            singles[x] = _gaussian_sampler(cfg.noise_scale)
        else:
            choice = str(rng.choice(nongaussian_names))
            singles[x] = _NONGAUSSIAN_FACTORIES[choice](cfg.noise_scale)

    return singles


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _default_name(cfg: RandomSCMConfig) -> str:
    return (
        f"random_n{cfg.n_nodes}_k{cfg.degree}_{cfg.linearity}_{cfg.noise}_s{cfg.seed}"
    )


def sample_random_scm_dataset(cfg: RandomSCMConfig) -> SCMDataset:
    """
    Sample a random SCM and return a ready-to-use :class:`SCMDataset`.

    The returned dataset produces exactly the same on-disk format as the hand-authored
    datasets in ``datasets.py`` when ``generate_ds`` is called (``ds.npz``, attention
    masks, metadata, graph), and can therefore be consumed by the existing data path
    unchanged.

    The full :class:`RandomSCMConfig` (including the seed) is stored under
    ``dataset.meta["random_scm_config"]`` so the dataset is exactly regenerable.

    Parameters
    ----------
    cfg : RandomSCMConfig
        Sampling configuration.

    Returns
    -------
    SCMDataset
        A dataset whose SCM was sampled according to ``cfg``.
    """
    rng = np.random.default_rng(int(cfg.seed))

    # 1) DAG structure (S/X roles + parents)
    names, source_labels, input_labels, parents = _sample_dag(cfg, rng)

    # 2) Structural equations + numeric params
    specs, params = _build_specs_and_params(cfg, names, source_labels, parents, rng)

    # 3) Per-node noise samplers
    singles = _build_singles(cfg, source_labels, input_labels, rng)

    # 4) Assemble the SCMDataset (target_labels intentionally empty)
    name = cfg.name or _default_name(cfg)
    n_edges = sum(len(p) for p in parents.values())
    description = (
        f"Randomly sampled SCM. n_nodes={cfg.n_nodes}, degree(k)={cfg.degree} "
        f"(target edges={round(cfg.degree * cfg.n_nodes)}, actual={n_edges}), "
        f"linearity={cfg.linearity}, noise={cfg.noise}, seed={cfg.seed}."
    )
    tags = ["random", cfg.linearity, cfg.noise, f"ER{cfg.degree}"]

    dataset = SCMDataset(
        name=name,
        description=description,
        tags=tags,
        specs=specs,
        params=params,
        singles=singles,
        groups=None,
        source_labels=source_labels,
        input_labels=input_labels,
        target_labels=[],
    )

    # 5) Persist the config in metadata for exact regeneration.
    cfg_dict = asdict(cfg)
    dataset.meta["random_scm_config"] = cfg_dict
    dataset.meta["n_edges"] = n_edges

    return dataset


__all__ = ["RandomSCMConfig", "sample_random_scm_dataset"]
