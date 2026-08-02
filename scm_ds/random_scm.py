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
(adjacency, attention masks, metadata, normalization, splits, graph visualization).
Hence this module only *builds ingredients* and returns an `SCMDataset`; `scm.py` is
left untouched.

Graph model: ER-k, exactly as in the benchmark literature
---------------------------------------------------------
The sampler reproduces the recipe used by NOTEARS (`simulate_dag`), DAGMA, gCastle
(`DAG.erdos_renyi`) and pcalg (`randDAG`):

1. draw a uniformly random topological order over the `n` nodes;
2. draw *exactly* `m = round(degree * n_nodes)` edges uniformly without replacement
   from the `C(n, 2)` forward slots (ER1 -> n edges, ER2 -> 2n, ...);
3. acyclicity holds by construction (every edge goes forward in the order).

Where do the sources (S) come from?
-----------------------------------
Nowhere in the literature is the number of root (source) nodes an input parameter for
ER/SF graphs: it is an *emergent* property of the sampled graph. We follow that
convention exactly - after sampling, every node with in-degree 0 is labelled `S` and
every other node is labelled `X`. Consequences:

- "no edge points into an S node" holds *by construction*, which is precisely the
  S->X / X->X block structure the cross/self-attention masks rely on;
- every X node has at least one parent, by definition;
- the number of sources scales with the graph, matching ER-k statistics:

      E[#roots] = sum_{i=0}^{n-1} (1-p)^i = (1 - (1-p)^n) / p ,  p = 2k / (n-1)
                ~= n * (1 - exp(-2k)) / (2k)

  i.e. ~43% of the nodes for ER1, ~24% for ER2 and ~12% for ER4 (verified by Monte
  Carlo). The analytic value is stored in the dataset metadata next to the observed
  one, so any generated dataset can be checked against the ER-k reference.

Label permutation
-----------------
Node names are assigned by a random permutation (`permute_labels=True`, default) so
that the variable *index* carries no information about the topological rank. Without
it, `X3` could only ever have parents among `X1, X2`, which is an ordering artefact a
baseline could exploit (cf. Reisach et al., "Beware of the Simulated DAG!").

Key properties
--------------
- Reproducibility: a single `seed` drives DAG sampling, structural-equation choices,
  weight sampling, label permutation and per-node noise-type choices. Same config +
  seed => identical SCM.
- Variance control: with `rescale_by_indegree=True` each parent weight is baked as
  `w / sqrt(in_degree)`, keeping per-node signal variance roughly stable as DAGs grow.

The full `RandomSCMConfig` is stored in the dataset's `meta` so any generated dataset
is exactly regenerable from its `meta.json`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
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
    Configuration describing a random ER-k SCM/DAG to sample.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes in the DAG (sources S + inputs X).
    degree : float
        ER-k expected number of edges *per node*. The graph gets exactly
        ``m = round(degree * n_nodes)`` edges.
    seed : int
        Master seed. Same config + seed always yields the same SCM/DAG.
    linearity : {"linear", "nonlinear", "mixed"}
        Structural-equation family. "mixed" chooses per-node.
    noise : {"gaussian", "nongaussian", "mixed"}
        Noise family for the X (input) nodes. "mixed" chooses per-node.
    weight_range : Tuple[float, float]
        Range for the *magnitude* of structural weights; sign is random.
    noise_scale : float
        Scale applied to X-node noise (sources use bounded ``source_noise``).
    source_noise : {"uniform", "gaussian"}
        Exogenous noise for the S sources. "uniform" gives bounded support so that
        do(S=.) interventions stay in-distribution (matches existing datasets).
    nonlinear_fns : Tuple[str, ...]
        Pool of nonlinear link functions to draw from. Supported:
        "square", "cube", "sin", "tanh".
    rescale_by_indegree : bool
        If True, bake ``w / sqrt(in_degree)`` into weights (variance control).
    permute_labels : bool
        If True (default), node names are assigned by a random permutation so the
        variable index reveals nothing about the topological order.
    name : Optional[str]
        Optional dataset name; a descriptive default is generated if omitted.

    Notes
    -----
    There is deliberately **no** ``n_sources`` knob: the sources are the roots of the
    sampled ER-k graph (see the module docstring).
    """
    n_nodes: int
    degree: float
    seed: int
    linearity: str = "linear"
    noise: str = "gaussian"
    weight_range: Tuple[float, float] = (0.5, 2.0)
    noise_scale: float = 0.1
    source_noise: str = "uniform"
    nonlinear_fns: Tuple[str, ...] = ("square", "cube", "sin", "tanh")
    rescale_by_indegree: bool = True
    permute_labels: bool = True
    name: Optional[str] = None


# --------------------------------------------------------------------------- #
# DAG sampling
# --------------------------------------------------------------------------- #

def expected_er_roots(n_nodes: int, degree: float) -> float:
    """
    Analytic expected number of root nodes of an ER-k DAG.

    With ``m = degree * n`` edges placed uniformly over the ``C(n, 2)`` forward slots
    of a random topological order, the node at position ``i`` has in-degree
    ``Binomial(i, p)`` with ``p = 2 * degree / (n - 1)``, hence

        E[#roots] = sum_{i=0}^{n-1} (1-p)^i = (1 - (1-p)^n) / p .

    Args:
        n_nodes: Number of nodes.
        degree: ER-k expected number of edges per node.

    Returns:
        Expected number of parentless nodes (float).
    """
    n = int(n_nodes)
    if n <= 1:
        return float(n)
    p = 2.0 * float(degree) / (n - 1)
    if p <= 0.0:
        return float(n)
    p = min(p, 1.0)
    return float((1.0 - (1.0 - p) ** n) / p)


def _sample_dag(
    cfg: RandomSCMConfig, rng: np.random.Generator
) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    """
    Sample an ER-k DAG and derive the S/X partition from its roots.

    Positions ``0 .. n-1`` are a random topological order. Exactly
    ``m = round(degree * n)`` edges are drawn uniformly without replacement from the
    forward slots ``(p, c)`` with ``p < c``, so acyclicity is guaranteed. Nodes with
    in-degree 0 become sources ``S``; all others become inputs ``X``.

    Names are assigned so that (a) sources are ``S1..Sa`` and inputs ``X1..Xb``, and
    (b) when ``permute_labels`` is set the numbering is a random permutation, hiding
    the topological order.

    Returns:
        ``(names, source_labels, input_labels, parents)`` where *names* lists the
        nodes in topological order, and *parents* maps each node name to its ordered
        list of parent names.
    """
    n = int(cfg.n_nodes)
    if n < 2:
        raise ValueError("n_nodes must be >= 2.")
    if cfg.degree <= 0:
        raise ValueError("degree must be > 0.")

    # ---- 1) edges: exactly m forward slots, uniformly without replacement ----
    n_slots = n * (n - 1) // 2
    target_m = int(round(float(cfg.degree) * n))
    m = int(min(max(target_m, 0), n_slots))

    # Slot index -> (parent_pos, child_pos), computed without materialising the
    # full C(n, 2) list (n can be ~1000 => 500k slots, still fine, but this keeps
    # memory flat and sampling exact).
    flat = rng.choice(n_slots, size=m, replace=False)
    # Slots are enumerated child-major: child c contributes c slots (parents 0..c-1),
    # so the offset of child c is c*(c-1)/2.
    child_pos = np.floor((1.0 + np.sqrt(1.0 + 8.0 * flat)) / 2.0).astype(np.int64)
    # Correct rare floating-point off-by-one at the boundaries.
    offset = child_pos * (child_pos - 1) // 2
    too_big = offset > flat
    child_pos[too_big] -= 1
    offset = child_pos * (child_pos - 1) // 2
    too_small = (offset + child_pos) <= flat
    child_pos[too_small] += 1
    offset = child_pos * (child_pos - 1) // 2
    parent_pos = flat - offset

    # ---- 2) roots (in-degree 0) are the sources ----
    has_parent = np.zeros(n, dtype=bool)
    has_parent[child_pos] = True
    source_pos = np.flatnonzero(~has_parent)
    input_pos = np.flatnonzero(has_parent)

    if source_pos.size == 0 or input_pos.size == 0:
        raise ValueError(
            f"Degenerate DAG: {source_pos.size} sources / {input_pos.size} inputs for "
            f"n_nodes={n}, degree={cfg.degree}. Lower the degree or raise n_nodes."
        )

    # ---- 3) names (optionally permuted so index != topological rank) ----
    s_ids = np.arange(1, source_pos.size + 1)
    x_ids = np.arange(1, input_pos.size + 1)
    if cfg.permute_labels:
        s_ids = rng.permutation(s_ids)
        x_ids = rng.permutation(x_ids)

    names: List[str] = [""] * n
    for slot, pos in enumerate(source_pos):
        names[int(pos)] = f"S{int(s_ids[slot])}"
    for slot, pos in enumerate(input_pos):
        names[int(pos)] = f"X{int(x_ids[slot])}"

    # Label lists are kept in ascending numeric order (S1, S2, ... / X1, X2, ...) so
    # the dataset column layout is independent of the sampled topology.
    source_labels = [f"S{i + 1}" for i in range(source_pos.size)]
    input_labels = [f"X{i + 1}" for i in range(input_pos.size)]

    # ---- 4) parent lists (parents sorted by topological position, for stability) ----
    parents: Dict[str, List[str]] = {nm: [] for nm in names}
    order = np.lexsort((parent_pos, child_pos))
    for i in order:
        parents[names[int(child_pos[i])]].append(names[int(parent_pos[i])])

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
    parents: Dict[str, List[str]],
    rng: np.random.Generator,
) -> Tuple[List[NodeSpec], Dict[str, float]]:
    """
    Build the NodeSpec list and the numeric params for the sampled DAG.

    Parentless nodes (the sources) are exogenous: ``expr = eps_<node>``. Every other
    node combines its parents (linear / nonlinear / mixed) plus its own noise.
    """
    specs: List[NodeSpec] = []
    params: Dict[str, float] = {}

    for node in names:
        node_parents = parents[node]

        if len(node_parents) == 0:
            specs.append(NodeSpec(node, [], f"eps_{node}"))
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
        f"random_n{cfg.n_nodes}_k{cfg.degree}_er_"
        f"{cfg.linearity}_{cfg.noise}_s{cfg.seed}"
    )


def sample_random_scm_dataset(cfg: RandomSCMConfig) -> SCMDataset:
    """
    Sample a random ER-k SCM and return a ready-to-use :class:`SCMDataset`.

    The returned dataset produces exactly the same on-disk format as the hand-authored
    datasets in ``datasets.py`` when ``generate_ds`` is called (``ds.npz``, attention
    masks, metadata, normalization, graph), and can therefore be consumed by the
    existing data path unchanged.

    The full :class:`RandomSCMConfig` (including the seed) is stored under
    ``dataset.meta["random_scm_config"]`` so the dataset is exactly regenerable, and
    the realised graph statistics under ``dataset.meta["graph_stats"]``.

    Args:
        cfg: Sampling configuration.

    Returns:
        An :class:`SCMDataset` whose SCM was sampled according to *cfg*.
    """
    rng = np.random.default_rng(int(cfg.seed))

    # 1) DAG structure; the S/X roles fall out of the sampled roots.
    names, source_labels, input_labels, parents = _sample_dag(cfg, rng)

    # 2) Structural equations + numeric params
    specs, params = _build_specs_and_params(cfg, names, parents, rng)

    # 3) Per-node noise samplers
    singles = _build_singles(cfg, source_labels, input_labels, rng)

    # 4) Assemble the SCMDataset (target_labels intentionally empty)
    name = cfg.name or _default_name(cfg)
    n_edges = sum(len(p) for p in parents.values())
    n_sources = len(source_labels)
    n_inputs = len(input_labels)
    expected_roots = expected_er_roots(cfg.n_nodes, cfg.degree)

    description = (
        f"Randomly sampled ER-{cfg.degree} SCM. n_nodes={cfg.n_nodes}, "
        f"edges={n_edges} (target={round(cfg.degree * cfg.n_nodes)}), "
        f"sources={n_sources} (ER expectation={expected_roots:.1f}), inputs={n_inputs}, "
        f"linearity={cfg.linearity}, noise={cfg.noise}, seed={cfg.seed}."
    )
    tags = ["random", "er", cfg.linearity, cfg.noise, f"ER{cfg.degree}"]

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

    # 5) Persist the config + realised graph statistics in metadata.
    dataset.meta["random_scm_config"] = asdict(cfg)
    dataset.meta["graph_stats"] = {
        "n_nodes": int(cfg.n_nodes),
        "n_edges": int(n_edges),
        "n_sources": int(n_sources),
        "n_inputs": int(n_inputs),
        "source_fraction": float(n_sources) / float(cfg.n_nodes),
        "expected_er_roots": float(expected_roots),
    }
    dataset.meta["n_edges"] = int(n_edges)

    return dataset


__all__ = ["RandomSCMConfig", "sample_random_scm_dataset", "expected_er_roots"]
