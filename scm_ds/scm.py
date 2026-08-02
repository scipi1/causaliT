"""
Minimal Structural Causal Model (SCM) with a symbolic authoring layer (SymPy) and a
compiled execution layer (NumPy). Supports:
- DAG validation (acyclicity via topological sort)
- Forward evaluation in topological order
- Vectorized sampling with node-specific noise
- Simple 'do()' interventions by clamping structural equations

Dependencies: sympy, numpy, pandas
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Any, Union
from pathlib import Path
import datetime
from os import makedirs
from os.path import join
import numpy as np
import pandas as pd
import sympy as sp
import json
from graphviz import Digraph
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ----------------------------- Type aliases --------------------------------- #

# A function that, given an RNG and number of samples, returns an array of noises.
NoiseSampler = Callable[[np.random.Generator, int], np.ndarray]

# A compiled structural mechanism: accepts parents (in declared order) and the node's noise.
CompiledMechanism = Callable[..., np.ndarray]

# Evaluation context mapping variable names to their realized arrays.
Context = MutableMapping[str, np.ndarray]



# ------------------------------- Noise model --------------------------------#

SingleSampler = Callable[[np.random.Generator, int], np.ndarray]          # → (n,)
GroupSampler  = Callable[[np.random.Generator, int], np.ndarray] 


@dataclass(frozen=True)
class GroupNoise:
    """Joint noise for a set of nodes; sampler returns an (n, k) matrix in the same node order."""
    nodes: Tuple[str, ...]
    sampler: GroupSampler

class NoiseModel:
    """
    Orchestrates exogenous noises: independent singles + correlated MVN groups.
    """
    def __init__(
        self,
        singles: Optional[Mapping[str, NoiseSampler]] = None,
        groups: Optional[Sequence[GroupNoise]] = None,
    ) -> None:
        self.singles: Dict[str, NoiseSampler] = dict(singles or {})
        self.groups: List[GroupNoise] = list(groups or [])
        # # Basic validation
        # seen = set()
        # for g in self.groups:
        #     k = len(g.nodes)
        #     if np.shape(g.mean) != (k,): raise ValueError("MVN mean shape mismatch.")
        #     if np.shape(g.cov)  != (k,k): raise ValueError("MVN cov shape mismatch.")
        #     if any(v in seen for v in g.nodes): raise ValueError("Node in multiple MVN groups.")
        #     seen.update(g.nodes)

    def sample_all(self, rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
        
        eps: Dict[str, np.ndarray] = {}
        # 1) Groups first (they may cover multiple nodes jointly)
        for g in self.groups:
            M = g.sampler(rng, n)  # shape (n, k)
            if M.ndim != 2 or M.shape[1] != len(g.nodes):
                raise ValueError("Group sampler must return (n, k) with k=len(nodes).")
            for j, v in enumerate(g.nodes):
                eps[v] = M[:, j]
        # 2) Singles for the rest
        for v, sampler in self.singles.items():
            if v not in eps:
                eps[v] = sampler(rng, n)
        return eps

# ------------------------------- Data model --------------------------------- #

@dataclass(frozen=True)
class NodeSpec:
    """
    Specification of a single SCM node (authoring layer).

    Attributes
    ----------
    name : str
        Node name (unique key in the graph), e.g., "Y".
    parents : List[str]
        Ordered list of parent node names, e.g., ["X", "Z"].
    expr : str
        Structural equation expressed as a SymPy-compatible string using:
          - Parent variable names as symbols (e.g., 'X', 'Z')
          - A node-specific noise symbol 'eps_<name>' (e.g., 'eps_Y')
        Example: "X + 2*Z + 0.1*eps_Y"
    """
    name: str
    parents: List[str]
    expr: str


# --------------------------------- Engine ----------------------------------- #

class SCM:
    """
    Structural Causal Model engine (execution layer).

    Parameters
    ----------
    specs : Sequence[NodeSpec]
        Collection of node specifications defining the DAG and equations.
    noise_samplers : Optional[Mapping[str, NoiseSampler]]
        Optional mapping from node name to a noise sampler. If omitted, each node
        receives i.i.d. standard Normal noise via `rng.standard_normal(n)`.
        To induce *latent confounding*, provide correlated samplers for selected nodes.
    backend : str, default "numpy"
        SymPy lambdify backend. Common options: "numpy", "jax", "numexpr" (where applicable).

    Notes
    -----
    - Acyclicity is enforced at initialization via topological sorting.
    - Each node v uses a dedicated noise symbol 'eps_v' inside its expression.
    - Sampling is vectorized: all node arrays are shape-(n,) by default.
    - Interventions via `do({...})` replace a node's structural mechanism with a constant.
    """

    def __init__(
        self,
        specs: Sequence[NodeSpec],
        #noise_samplers: Optional[Mapping[str, NoiseSampler]] = None,
        noise_model: Optional[NoiseModel] = None,
        backend: str = "numpy",
    ) -> None:
        # Store specifications by name
        self.specs: Dict[str, NodeSpec] = {s.name: s for s in specs}

        
        # Validate and cache topological order
        self.order: List[str] = self._topo_order()

        # Create a noise symbol per node: eps_<name>
        self._eps_sym: Dict[str, sp.Symbol] = {v: sp.symbols(f"eps_{v}") for v in self.specs}

        # Compile each node's symbolic expression into a vectorized callable
        self._fns: Dict[str, CompiledMechanism] = {}
        for v, s in self.specs.items():
            # Build a local symbol table for parents and the node's own noise
            parent_symbols = {p: sp.symbols(p) for p in s.parents}
            local_symbols = {**parent_symbols, f"eps_{v}": self._eps_sym[v]}
            expr_sym = sp.sympify(s.expr, locals=local_symbols)
            # Order of arguments: parents..., eps_v
            self._fns[v] = sp.lambdify([*parent_symbols.values(), self._eps_sym[v]], expr_sym, backend)

        # Default: i.i.d. standard normal noise if no samplers provided
        if noise_model is None:
            self.noise: Dict[str, NoiseSampler] = {
                v: (lambda rng, n: rng.standard_normal(n)) for v in self.specs
            }
        else:
            # Copy into a mutable dict so that do()-derived copies can re-use it
            self.noise = noise_model

    # ----------------------------- Internals -------------------------------- #

    def _topo_order(self) -> List[str]:
        """
        Compute a topological order of the DAG defined by `self.specs`.

        Returns
        -------
        order : List[str]
            Node names in a valid evaluation order.

        Raises
        ------
        ValueError
            If a cycle is detected (i.e., the graph is not a DAG).
        """
        indeg: Dict[str, int] = defaultdict(int)
        graph: Dict[str, List[str]] = defaultdict(list)

        # Build adjacency and indegree
        for s in self.specs.values():
            for p in s.parents:
                graph[p].append(s.name)
                indeg[s.name] += 1
            indeg.setdefault(s.name, 0)

        # Kahn's algorithm
        q: deque[str] = deque([v for v, d in indeg.items() if d == 0])
        order: List[str] = []
        while q:
            u = q.popleft()
            order.append(u)
            for w in graph[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)

        if len(order) != len(self.specs):
            raise ValueError("Graph has a cycle; SCM requires a DAG.")
        return order

    @staticmethod
    def _constant_fn(value: float) -> CompiledMechanism:
        """
        Create a mechanism that ignores inputs and returns a broadcasted constant.

        The function signature matches compiled mechanisms: (*parents, eps_v) -> array.
        The output shape is inferred from the shape of the last argument (node noise),
        which is always present in `forward()` calls.

        Parameters
        ----------
        value : float
            Constant value to broadcast.

        Returns
        -------
        CompiledMechanism
            Callable that returns an array filled with `value`, with shape matching eps_v.
        """
        def fn(*args: np.ndarray) -> np.ndarray:
            # args contain [parents..., eps_v]; infer batch shape from eps_v
            shape = np.shape(args[-1])
            return np.full(shape, value)
        return fn

    # ------------------------------ API ------------------------------------- #

    def forward(self, context: Context, eps_draws: Mapping[str, np.ndarray]) -> Context:
        """
        Evaluate all nodes in topological order and fill `context` in-place.

        Parameters
        ----------
        context : MutableMapping[str, np.ndarray]
            Mapping from node name to realized arrays. Will be populated for every node.
        eps_draws : Mapping[str, np.ndarray]
            Node-specific noise arrays, typically shape (n,).

        Returns
        -------
        Context
            The same `context` mapping, filled with computed node arrays.
        """
        
        for v in self.order:
            parents = self.specs[v].parents
            args = [context[p] for p in parents] + [eps_draws[v]]
            context[v] = self._fns[v](*args)
        return context

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        """
        Draw observational samples from the SCM.

        Parameters
        ----------
        n : int
            Number of i.i.d. units to sample.
        seed : int, default 0
            Seed for the RNG (np.random.default_rng) for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with one column per node and `n` rows.
        """
        rng = np.random.default_rng(seed)
        if isinstance(self.noise, dict):
            eps_draws: Dict[str, np.ndarray] = {v: self.noise[v](rng, n) for v in self.specs}
        else:
            eps_draws: NoiseModel = self.noise.sample_all(rng, n)
        
        ctx: Context = {}
        self.forward(ctx, eps_draws)
        # Ensure 1D shape (n,) for each column
        return pd.DataFrame({k: np.asarray(v).reshape(n) for k, v in ctx.items()})

    def do(self, interventions: Mapping[str, float]) -> SCM:
        """
        Return an intervened SCM where selected nodes are clamped to constants.

        This implements Pearl's do-operator by replacing the node's structural
        equation f_v with a constant function returning the intervened value.

        Parameters
        ----------
        interventions : Mapping[str, float]
            Mapping node -> constant value, e.g., {"Z": 1.0}.

        Returns
        -------
        SCM
            A new SCM instance sharing the same DAG and noise samplers but with the
            specified nodes clamped.

        Notes
        -----
        - Parents are kept for logging/metadata, but their values are ignored by
          the clamped mechanism; the mechanism's output shape matches the node's noise.
        - Downstream nodes will reflect the intervention when sampling.
        """
        # Shallow "copy": reuse specs and noise; recompile fns and then override targets
        new = SCM(list(self.specs.values()), noise_model=self.noise)
        for v, val in interventions.items():
            if v not in new.specs:
                raise KeyError(f"Intervention targets unknown node '{v}'.")
            new._fns[v] = self._constant_fn(val)
        # Reuse the same topological order (same DAG)
        new.order = self.order
        return new
    
    
    def edges(self) -> List[Tuple[str, str]]:
        """
        Return the **directed** edge list implied by NodeSpecs as (parent, child) tuples.
        """
        E: List[Tuple[str, str]] = []
        for child, spec in self.specs.items():
            for parent in spec.parents:
                E.append((parent, child))
        return E


    def adjacency(
        self,
        nodes: Optional[Sequence[str]] = None,
        positive_child: bool = True,
        as_dataframe: bool = True,
    ):
        """
        Build the adjacency matrix A (|V| x |V|) from NodeSpecs.
        N.B. A is built with 1 where the edge is entering! This is useful to define
        masks in attention blocks: the node can see only its parents
        Parameters
        ----------
        nodes : optional explicit node order for rows/cols; if None,
                uses self.order (topological). If you need a fixed, public order
                (e.g., for metadata), pass it explicitly.
        positive_child : if True, incoming edges are set to 1
        as_dataframe : if True, return a pandas.DataFrame with labeled axes.

        Returns
        -------
        A : np.ndarray or pd.DataFrame
            A[i, j] = 1 if nodes[i] <- nodes[j], else 0.
        """
        node_list: List[str] = list(nodes) if nodes is not None else list(self.order)
        idx: Dict[str, int] = {v: i for i, v in enumerate(node_list)}
        A = np.zeros((len(node_list), len(node_list)), dtype=int)
        for u, v in self.edges():
            if u in idx and v in idx:
                if positive_child:
                    A[idx[v], idx[u]] = 1
                else:
                    A[idx[u], idx[v]] = 1
            else:
                # If a node is missing from `nodes`, you may want to raise instead:
                # raise KeyError(f"Edge ({u}->{v}) references node not in provided `nodes`.")
                pass

        if as_dataframe:
            import pandas as pd
            return pd.DataFrame(A, index=node_list, columns=node_list)
        return A


    def to_graphviz(
        self,
        *,
        rankdir: str = "LR",
        node_attrs: Optional[Dict[str, str]] = None,
        edge_attrs: Optional[Dict[str, str]] = None,
    ):
        """
        Create a Graphviz Digraph from the SCM.

        Parameters
        ----------
        rankdir : "LR" (left-right) or "TB" (top-bottom).
        node_attrs : default attrs for all nodes (e.g., {"shape":"box"}).
        edge_attrs : default attrs for all edges.

        Returns
        -------
        graphviz.Digraph
            Use .source for DOT text, .render()/ .save() to write files.
        """
        if Digraph is None:
            raise ImportError(
                "graphviz package is not available. Install with `pip install graphviz` "
                "and ensure the Graphviz system binary (dot) is on PATH."
            )

        g = Digraph(
            name="SCM",
            graph_attr={"rankdir": rankdir},
            node_attr=node_attrs or {},
            edge_attr=edge_attrs or {},
            format="svg",
        )
        # Stable order: topological by default
        for v in self.order:
            g.node(v, v)
        for u, v in self.edges():
            g.edge(u, v)
        return g
    
    
@dataclass(frozen=True)
class Spec:
    name: str
    nodes: List[NodeSpec]                   # from earlier class
    params: Dict[str, Any] = field(default_factory=dict)  # free hyperparams (weights, scales)
    
    # Optional noise inputs:
    noise_model: Optional[NoiseModel] = None                      # <- pass-through if set
    single_noises: Dict[str, SingleSampler] = field(default_factory=dict)
    group_noises: List[GroupNoise] = field(default_factory=list)
    noise_scales: Dict[str, float] = field(default_factory=dict)  # only used for fallback defaults


    def validate(self) -> None:
        """Run sanity checks: DAG acyclicity, domains, overlap, SNR, etc."""
        
        print("Verifying specs ...")
        
        # 1) acyclicity via a temporary SCM(self.nodes). _topo_order()
        if isinstance(SCM(self.nodes)._topo_order(), list):
            print("DAG is acyclic ✓")
            
        # 2) check noise specs cover all exogenous nodes uniquely or via MVN group
        # 3) optional: quick Monte Carlo probe for propensity ∈ (ε,1−ε), value ranges
        
    def _baked_nodes(self) -> List[NodeSpec]:
        """
        Substitute numeric `params` into node expressions, returning a new NodeSpec list
        with **purely numeric** expressions (ready for SCM).
        """
        const_symbols = {k: sp.symbols(k) for k in self.params.keys()}
        baked: List[NodeSpec] = []
        for s in self.nodes:
            # parents + eps_<name> + constants available to sympify
            local = {**{p: sp.symbols(p) for p in s.parents},
                     f"eps_{s.name}": sp.symbols(f"eps_{s.name}"),
                     **const_symbols}
            expr_sym = sp.sympify(s.expr, locals=local).subs(self.params)
            baked.append(NodeSpec(name=s.name, parents=s.parents, expr=str(expr_sym)))
        return baked


    def to_scm(self, backend: str = "numpy") -> SCM:
        """Compile this Spec into an executable SCM with correct noise samplers."""
        # build noise_samplers from self.noises (MVN sampler for grouped nodes)
        return SCM(self._baked_nodes(), self.noise_model, backend=backend)
    
    
    
class SCMDataset:
    def __init__(self,
    
        # Description
        name: str,
        description: str,
        tags: List[str],

        # Structural equations
        specs: List[NodeSpec],
        params: Dict[str, float],

        # Noise
        singles : Dict[str, float],
        groups  : List[GroupNoise],
        
        # Dataset info
        input_labels: List[str],
        target_labels: List[str],
        source_labels: Optional[List[str]] = None,
    ) -> None:
        
        self.noise_model = NoiseModel(singles=singles, groups=groups)
        self.specs = specs  # Store original specs for metadata generation
        specs_scm = Spec(
            name=name,
            nodes=specs,
            params=params,
            noise_model=self.noise_model,
        )
        self.scm = specs_scm.to_scm()
        self.input_labels = input_labels
        self.target_labels = target_labels
        self.source_labels = source_labels
        self.name = name
        self.description = description
        self.tags = tags
        
        # meta
        created = datetime.date.today().isoformat()
        self.meta: Dict[str, Any] = {
            "name": name,
            "created": created,
            "description": description,
        }
        
        
    def sample(self, n, seed=42):
        return self.scm.sample(n, seed)
    
    def _compute_transitive_closure(self) -> Dict[str, List[str]]:
        """
        Compute the transitive closure (reachability) from each source node.
        
        Returns a dict mapping each source variable to a list of all variables 
        it can reach (directly or indirectly).
        """
        # Build adjacency list from specs: parent -> [children]
        children_of: Dict[str, List[str]] = defaultdict(list)
        for spec in self.specs:
            for parent in spec.parents:
                children_of[parent].append(spec.name)
        
        # Compute reachability using BFS from each source
        all_sources = self.source_labels if self.source_labels else []
        transitive_closure: Dict[str, List[str]] = {}
        
        for source in all_sources:
            reachable = set()
            queue = deque([source])
            visited = {source}
            
            while queue:
                node = queue.popleft()
                for child in children_of.get(node, []):
                    reachable.add(child)
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
            
            transitive_closure[source] = sorted(list(reachable))
        
        return transitive_closure
    
    def _compute_expected_effects(self) -> Dict[str, Dict[str, bool]]:
        """
        Compute expected causal effects from each source to each input/target variable.
        
        Based on transitive closure: if source S can reach variable X, then 
        do(S=x) should affect X.
        """
        transitive_closure = self._compute_transitive_closure()
        
        # Get all target variables for intervention effects
        all_targets = list(self.input_labels) + list(self.target_labels)
        all_sources = self.source_labels if self.source_labels else []
        
        expected_effects: Dict[str, Dict[str, bool]] = {}
        
        for source in all_sources:
            reachable = set(transitive_closure.get(source, []))
            expected_effects[source] = {
                target: (target in reachable)
                for target in all_targets
            }
        
        return expected_effects
    
    def compute_interventional_expectation(
        self, 
        intervention: Dict[str, float],
        target_vars: Optional[List[str]] = None,
        n_samples: int = 10000,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Compute E[X | do(S=s)] for each target variable using Monte Carlo sampling.
        
        This computes the expected value of downstream variables under a do-intervention.
        For synthetic SCMs, this provides the ground-truth causal effect.
        
        Parameters
        ----------
        intervention : Dict[str, float]
            Mapping from variable name to intervention value, e.g., {"S1": 0.5}
        target_vars : Optional[List[str]]
            List of target variable names to compute expectations for.
            If None, uses all input_labels.
        n_samples : int
            Number of samples for Monte Carlo estimation
        seed : int
            Random seed for Monte Carlo sampling
            
        Returns
        -------
        Dict[str, float]
            Mapping from variable name to expected value under intervention
            
        Example
        -------
        >>> scm_ds.compute_interventional_expectation(
        ...     intervention={"S2": 0.5},
        ...     target_vars=["X1", "X2"],
        ... )
        {"X1": 0.5, "X2": 0.0}  # X1 depends on S2, X2 does not
        """
        if target_vars is None:
            target_vars = list(self.input_labels)
        
        return self._compute_interventional_expectation_mc(
            intervention, target_vars, n_samples, seed
        )
    
    def _compute_interventional_expectation_mc(
        self,
        intervention: Dict[str, float],
        target_vars: List[str],
        n_samples: int,
        seed: int
    ) -> Dict[str, float]:
        """
        Compute interventional expectations using Monte Carlo sampling.
        
        Samples from the intervened SCM and computes empirical mean.
        """
        scm_do = self.scm.do(intervention)
        df_samples = scm_do.sample(n=n_samples, seed=seed)
        
        expectations = {}
        for var in target_vars:
            if var in df_samples.columns:
                expectations[var] = float(df_samples[var].mean())
            else:
                # Variable might be in source_labels or not exist
                if var in intervention:
                    expectations[var] = intervention[var]
                else:
                    expectations[var] = float('nan')
        
        return expectations
    
    def compute_ate_ground_truth(
        self,
        do_values: Union[List[float], Dict[str, List[float]]] = None,
        n_samples: int = 50000,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Compute ground-truth ATE for all source -> input variable combinations
        using Monte Carlo sampling.
        
        ATE is defined as the DIFFERENCE:
            ATE(S→X, s) = E[X | do(S=s)] - E[X | do(S=0)]
        
        Parameters
        ----------
        do_values : Union[List[float], Dict[str, List[float]]]
            Either:
            - List[float]: Same values applied to all source variables
            - Dict[str, List[float]]: Per-source intervention values, e.g.,
              {"S1": [0.5], "S2": [-1.7], "S3": [-0.5, 1.0]}
            Default: PAPER_INTERVENTIONS (paper-specific values)
        n_samples : int
            Number of samples for Monte Carlo estimation (default 50,000)
        seed : int
            Random seed
            
        Returns
        -------
        Dict with keys:
            - "ate": Dict[str, Dict[str, float]] - ATE values (treated - baseline)
            - "baseline": Dict[str, float] - E[X | do(S=0)] for each X
            - "treated": Dict[str, Dict[str, float]] - E[X | do(S=s)] for each intervention
            
        Example
        -------
        >>> ate_gt = scm_ds.compute_ate_ground_truth()
        >>> ate_gt["ate"]["S1=0.5"]["X1"]  # ATE = E[X1|do(S1=0.5)] - E[X1|do(S1=0)] = 0
        """
        # Paper-specific intervention values (unified for discrete & continuous):
        # - S1: dangling (no children) - negative control
        # - S2: one-to-one → X1 → X5 - positive control
        # - S3: one-to-many → X2, X3; propagates to X4, X5 - structure test
        # - S5: many-to-one → X4 - confounding test
        # ID values: ±0.5 (safely inside continuous support [-1,1])
        # OOD value: 1.5 (clearly outside [-1,1], not in any discrete support)
        PAPER_INTERVENTIONS = {
            "S1": [0.5],           # ID only (dangling, null effect expected)
            "S2": [0.5, 1.5],     # ID (0.5) + OOD (1.5) - positive control
            "S3": [-0.5, 1.5],    # ID (-0.5) + OOD (1.5) - structure test
            "S5": [-0.5, 1.5],    # ID (-0.5) + OOD (1.5) - confounding test
        }
        
        if do_values is None:
            do_values = PAPER_INTERVENTIONS
        
        if not self.source_labels:
            raise ValueError("No source_labels defined. Cannot compute ATE ground truth.")
        
        # Compute baseline: E[X | do(S=0)] for each source
        # For ATE, baseline is do(S=0) for the specific source being intervened
        baseline_expectations: Dict[str, Dict[str, float]] = {}
        for source in self.source_labels:
            baseline_intervention = {source: 0.0}
            baseline_expectations[source] = self.compute_interventional_expectation(
                intervention=baseline_intervention,
                target_vars=self.input_labels,
                n_samples=n_samples,
                seed=seed
            )
        
        ate_values: Dict[str, Dict[str, float]] = {}
        treated_expectations: Dict[str, Dict[str, float]] = {}
        
        # Handle both Dict and List formats
        if isinstance(do_values, dict):
            # Per-source intervention values
            for source, values in do_values.items():
                if source not in self.source_labels:
                    continue  # Skip sources not in this dataset
                for do_value in values:
                    intervention = {source: do_value}
                    key = f"{source}={do_value}"
                    
                    # Compute E[X | do(S=s)]
                    treated = self.compute_interventional_expectation(
                        intervention=intervention,
                        target_vars=self.input_labels,
                        n_samples=n_samples,
                        seed=seed
                    )
                    treated_expectations[key] = treated
                    
                    # Compute ATE = treated - baseline
                    baseline = baseline_expectations[source]
                    ate_values[key] = {
                        var: treated[var] - baseline[var]
                        for var in self.input_labels
                    }
        else:
            # Same values for all sources (legacy behavior)
            for source in self.source_labels:
                for do_value in do_values:
                    intervention = {source: do_value}
                    key = f"{source}={do_value}"
                    
                    # Compute E[X | do(S=s)]
                    treated = self.compute_interventional_expectation(
                        intervention=intervention,
                        target_vars=self.input_labels,
                        n_samples=n_samples,
                        seed=seed
                    )
                    treated_expectations[key] = treated
                    
                    # Compute ATE = treated - baseline
                    baseline = baseline_expectations[source]
                    ate_values[key] = {
                        var: treated[var] - baseline[var]
                        for var in self.input_labels
                    }
        
        return {
            "ate": ate_values,
            "baseline": baseline_expectations,
            "treated": treated_expectations,
        }
    
    def _generate_dataset_metadata(self, shared_vars_map: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive dataset metadata for evaluation functions.
        
        This metadata enables evaluation functions to be dataset-agnostic by providing:
        - Variable information (labels, counts, index mappings)
        - Causal structure (edges, transitive closure, expected effects)
        
        Returns:
            Dict containing all metadata needed by evaluation functions.
        """
        # Compute direct edges from specs
        direct_edges = []
        for spec in self.specs:
            for parent in spec.parents:
                direct_edges.append([parent, spec.name])
        
        # Compute transitive closure and expected effects
        transitive_closure = self._compute_transitive_closure()
        expected_effects = self._compute_expected_effects()
        
        # Build variable index map
        if shared_vars_map is not None:
            variable_index_map = shared_vars_map
        else:
            # Build from individual label lists
            variable_index_map = {}
            idx = 1
            if self.source_labels:
                for var in self.source_labels:
                    variable_index_map[var] = idx
                    idx += 1
            for var in self.input_labels:
                variable_index_map[var] = idx
                idx += 1
            for var in self.target_labels:
                variable_index_map[var] = idx
                idx += 1
        
        metadata = {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "variable_info": {
                "source_labels": self.source_labels if self.source_labels else [],
                "input_labels": list(self.input_labels),
                "target_labels": list(self.target_labels),
                "n_source": len(self.source_labels) if self.source_labels else 0,
                "n_input": len(self.input_labels),
                "n_target": len(self.target_labels),
            },
            "causal_structure": {
                "direct_edges": direct_edges,
                "transitive_closure": transitive_closure,
                "expected_effects": expected_effects,
            },
            "variable_index_map": variable_index_map,
            # Feature indices define which column in the data tensor corresponds to which feature
            # These are used for embedding configuration (value_idx, var_idx in OrthogonalMaskEmbedding)
            # For SCM datasets: always {value: 0, variable: 1}
            # For real-world datasets: may vary (e.g., dyconex has additional descriptive features)
            "feature_indices": {
                "value": 0,
                "variable": 1,
            },
        }
        
        return metadata
    
    def _apply_split(
        self,
        input_np: np.ndarray,
        target_np: Optional[np.ndarray],
        source_np: Optional[np.ndarray],
        test_split_method: Dict[str, Any],
        sv_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Apply train/test split based on the specified method.
        
        Parameters
        ----------
        input_np : np.ndarray
            Input data array of shape (n_samples, seq_len, features)
        target_np : Optional[np.ndarray]
            Target data array (can be None if no targets)
        source_np : Optional[np.ndarray]
            Source data array (can be None if no sources)
        test_split_method : Dict[str, Any]
            Split configuration with "method" and "kwargs" keys.
            Methods: "ratio" or "holdout"
        sv_map : Optional[Dict[str, int]]
            Source variable name to index mapping (needed for holdout method)
            
        Returns
        -------
        train_data : Dict[str, np.ndarray]
            Dictionary with "x", and optionally "y", "s" arrays for training
        test_data : Dict[str, np.ndarray]
            Dictionary with "x", and optionally "y", "s" arrays for testing
        split_info : Dict[str, Any]
            Metadata about the split (method, sizes, etc.)
        """
        method = test_split_method.get("method")
        kwargs = test_split_method.get("kwargs", {})
        n_total = input_np.shape[0]
        
        if method == "ratio":
            # Random ratio-based split
            train_ratio = kwargs.get("train_ratio", 0.8)
            if train_ratio < 0.8:
                raise ValueError(f"train_ratio must be >= 0.8 to ensure sufficient training data. Got {train_ratio}")
            
            n_train = int(n_total * train_ratio)
            
            # Create random permutation for shuffling
            rng = np.random.default_rng(kwargs.get("seed", 42))
            indices = rng.permutation(n_total)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            split_info = {
                "method": "ratio",
                "train_ratio": train_ratio,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "train_fraction": len(train_idx) / n_total,
            }
            
        elif method == "holdout":
            # Holdout split based on explicit S values
            explicit_values = kwargs.get("explicit_values", {})
            
            if not explicit_values:
                raise ValueError("holdout method requires 'explicit_values' in kwargs")
            
            if source_np is None or sv_map is None:
                raise ValueError("holdout method requires source data and sv_map")
            
            # Reverse sv_map: index -> variable name
            idx_to_var = {v: k for k, v in sv_map.items()}
            
            # Find samples that match ANY holdout value (OR logic)
            holdout_mask = np.zeros(n_total, dtype=bool)
            
            for var_name, holdout_vals in explicit_values.items():
                if var_name not in sv_map:
                    raise ValueError(f"Variable '{var_name}' not found in source variables")
                
                # Find which position in source_np corresponds to this variable
                var_idx = sv_map[var_name]
                # Find the sequence position where this variable appears
                # source_np shape: (n_samples, seq_len, features) where features[1] is variable index
                seq_len = source_np.shape[1]
                var_seq_pos = None
                for pos in range(seq_len):
                    if source_np[0, pos, 1] == var_idx:  # Check variable index in first sample
                        var_seq_pos = pos
                        break
                
                if var_seq_pos is None:
                    raise ValueError(f"Could not find sequence position for variable '{var_name}'")
                
                # Get values of this variable for all samples
                var_values = source_np[:, var_seq_pos, 0]  # Feature 0 is value
                
                # Mark samples that have any of the holdout values
                for holdout_val in holdout_vals:
                    # Use approximate matching due to floating point
                    matches = np.isclose(var_values, holdout_val, rtol=1e-5, atol=1e-8)
                    holdout_mask |= matches
            
            test_idx = np.where(holdout_mask)[0]
            train_idx = np.where(~holdout_mask)[0]
            
            # Validate minimum training data requirement (80%)
            train_fraction = len(train_idx) / n_total
            if train_fraction < 0.8:
                raise ValueError(
                    f"Holdout configuration results in only {train_fraction:.1%} training data "
                    f"({len(train_idx)} samples). Minimum required is 80%. "
                    f"Please reduce the number of holdout values."
                )
            
            split_info = {
                "method": "holdout",
                "explicit_values": explicit_values,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "train_fraction": train_fraction,
            }
            
        else:
            raise ValueError(f"Unknown split method: {method}. Use 'ratio' or 'holdout'.")
        
        # Apply split to all arrays
        train_data = {"x": input_np[train_idx]}
        test_data = {"x": input_np[test_idx]}
        
        if target_np is not None:
            train_data["y"] = target_np[train_idx]
            test_data["y"] = target_np[test_idx]
        
        if source_np is not None:
            train_data["s"] = source_np[train_idx]
            test_data["s"] = source_np[test_idx]
        
        # Print split info
        print(f"\n=== Train/Test Split ===")
        print(f"Method: {method}")
        print(f"Train samples: {split_info['n_train']} ({split_info['train_fraction']:.1%})")
        print(f"Test samples: {split_info['n_test']} ({1 - split_info['train_fraction']:.1%})")
        if method == "holdout":
            print(f"Holdout values: {explicit_values}")
        print("========================\n")
        
        return train_data, test_data, split_info
    
    def _normalize(self, data: np.ndarray, method: str = "standardize"):
        """Normalize only the value features (feature index 0) using sklearn scalers."""
        normalized_data = data.copy()
        values = data[:, :, 0].reshape(-1, 1)  # Flatten to (batch*seq, 1)
        
        if method == "standardize":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        normalized_values = scaler.fit_transform(values).reshape(data.shape[0], data.shape[1])
        normalized_data[:, :, 0] = normalized_values
        
        stats = {"method": method, "mean": float(scaler.mean_[0]) if hasattr(scaler, 'mean_') else None, 
                 "std": float(np.sqrt(scaler.var_[0])) if hasattr(scaler, 'var_') else None,
                 "min": float(scaler.data_min_[0]) if hasattr(scaler, 'data_min_') else None,
                 "max": float(scaler.data_max_[0]) if hasattr(scaler, 'data_max_') else None}
        
        return normalized_data, {k: v for k, v in stats.items() if v is not None}
    
    def get_numpy(self, mode, n, seed=42, shared_embedding=False):
        # reshape the dataset into B x L x D, where
                # - B: batch/sample size
                # - L: is the sequence length
                # - D: is the feature length (value and variable)
        
        df = self.sample(n,seed)
        
        if mode == "flat":
            
            # Build shared mapping if requested (order: source -> input -> target)
            if shared_embedding:
                all_labels = []
                if self.source_labels is not None:
                    all_labels.extend(self.source_labels)
                all_labels.extend(self.input_labels)
                if self.target_labels:  # Only add if non-empty
                    all_labels.extend(self.target_labels)
                shared_vars_map = {var: i+1 for i, var in enumerate(all_labels)}
            else:
                shared_vars_map = None
            
            def to_numpy_(label, vars_map_override=None):
                # Handle empty label list - return None values
                if not label:
                    return None, {}, {}, np.array([])
                
                df_ = pd.melt(df[label], id_vars=None, ignore_index=False)
                unique_vars = df_["variable"].unique()
                
                # convert vars --> int and save map
                if vars_map_override is not None:
                    # Use the shared mapping (filter to only include vars in this category)
                    vars_map = {var: vars_map_override[var] for var in unique_vars}
                else:
                    vars_map = {var : i+1 for i,var in enumerate(unique_vars)}
                df_["variable"] = df_["variable"].map(vars_map)
                
                feats = ['value', 'variable']
                feat_map = {i : feat for i, feat in enumerate(feats)}
                
                # reshape the dataframe to B x L x D
                g = df_.groupby(level=0)
                seq_vars = None
                arrays = []
                for _, group in g:
                    arr = group[feats].values
                    
                    # check that the variable order is the same for every sample
                    seq_vars_arr = group['variable'].values
                    if seq_vars is None:
                        seq_vars = seq_vars_arr
                    else:
                        assert seq_vars_arr.all() == seq_vars.all(), AssertionError("Inconsistent variable sequence!")
                        
                    arrays.append(arr)
                
                return np.stack(arrays, axis=0), vars_map, feat_map, seq_vars
            
            input_np, iv_map, if_map, iv_order = to_numpy_(self.input_labels, shared_vars_map)
            target_np, tv_map, tf_map, tv_order = to_numpy_(self.target_labels, shared_vars_map)
            
            # Handle source_labels if present
            if self.source_labels is not None:
                source_np, sv_map, sf_map, sv_order = to_numpy_(self.source_labels, shared_vars_map)
                return input_np, (iv_map, if_map, iv_order), target_np, (tv_map, tf_map, tv_order), source_np, (sv_map, sf_map, sv_order), shared_vars_map
            
            return input_np, (iv_map, if_map, iv_order), target_np , (tv_map, tf_map, tv_order), shared_vars_map

            
    def generate_ds(self, mode, n, save_dir: Union[str, Path]=None, meta_dict: dict=None,
                    normalize: bool = True, normalize_method: str = "standardize", seed=42,
                    shared_embedding: bool = False,
                    test_split_method: Optional[Dict[str, Any]] = None,
                    compute_ate: bool = True):

        
        # get numpy array - handle both cases (with and without source_labels)
        get_numpy_result = self.get_numpy(mode, n, seed, shared_embedding=shared_embedding)
        
        if self.source_labels is not None:
            # Unpack with source data
            input_np, (iv_map, if_map, iv_order), target_np, (tv_map, tf_map, tv_order), source_np, (sv_map, sf_map, sv_order), shared_vars_map = get_numpy_result
            print("numpy arrays generated (including source)")
        else:
            # Unpack without source data
            input_np, (iv_map, if_map, iv_order), target_np, (tv_map, tf_map, tv_order), shared_vars_map = get_numpy_result
            source_np, sv_map, sf_map, sv_order = None, None, None, None
            print("numpy arrays generated")
        
        # --------------------- Train/Test Split -----------------------
        # IMPORTANT: Apply split BEFORE normalization to match against original S values
        split_info = None
        train_data = None
        test_data = None
        
        if test_split_method is not None and test_split_method.get("method") is not None:
            train_data, test_data, split_info = self._apply_split(
                input_np=input_np,
                target_np=target_np,
                source_np=source_np,
                test_split_method=test_split_method,
                sv_map=sv_map,
            )
        
        # --------------------- Normalization -----------------------
        # Normalize after split (applies to both split and non-split cases)
        norm_stats = {}
        if normalize:
            if split_info is not None:
                # Normalize train and test data separately
                train_data["x"], input_stats = self._normalize(train_data["x"], method=normalize_method)
                test_data["x"], _ = self._normalize(test_data["x"], method=normalize_method)
                norm_stats = {"input": input_stats}
                
                if "y" in train_data:
                    train_data["y"], target_stats = self._normalize(train_data["y"], method=normalize_method)
                    test_data["y"], _ = self._normalize(test_data["y"], method=normalize_method)
                    norm_stats["target"] = target_stats
                
                if "s" in train_data:
                    train_data["s"], source_stats = self._normalize(train_data["s"], method=normalize_method)
                    test_data["s"], _ = self._normalize(test_data["s"], method=normalize_method)
                    norm_stats["source"] = source_stats
            else:
                # Original behavior: normalize full arrays
                input_np, input_stats = self._normalize(input_np, method=normalize_method)
                norm_stats = {"input": input_stats}
                
                if target_np is not None:
                    target_np, target_stats = self._normalize(target_np, method=normalize_method)
                    norm_stats["target"] = target_stats
                
                if source_np is not None:
                    source_np, source_stats = self._normalize(source_np, method=normalize_method)
                    norm_stats["source"] = source_stats
            
            # Print normalization stats
            print(f"Data normalized using {normalize_method}")
            print(f"  Input - mean: {input_stats.get('mean', 'N/A')}, std: {input_stats.get('std', 'N/A')}")
            if target_np is not None or (split_info and "y" in train_data):
                print(f"  Target - mean: {norm_stats.get('target', {}).get('mean', 'N/A')}, std: {norm_stats.get('target', {}).get('std', 'N/A')}")
            else:
                print(f"  Target - (no target variables)")
            if source_np is not None or (split_info and "s" in train_data):
                print(f"  Source - mean: {norm_stats.get('source', {}).get('mean', 'N/A')}, std: {norm_stats.get('source', {}).get('std', 'N/A')}")
        
        
        # ------------------ make attention masks -----------------------
        # Branching logic: if source_labels is None, create specific attention masks
        # Otherwise, export full DAG adjacency matrix
        
        if self.source_labels is None:
            df_adj = self.scm.adjacency(positive_child=True, as_dataframe=True)
            # Original behavior: create three separate attention masks
            # rows for queries, cols for keys
            
            # encoder self-attention
            df_esa = df_adj.loc[self.input_labels, self.input_labels]
            assert np.array_equal(df_esa.index.to_numpy(), df_esa.columns.to_numpy()) # self-attention: rows == cols
            assert np.array_equal(df_esa.index.map(iv_map).to_numpy(), iv_order)      # rows == input variable sequential order
            
            # decoder self-attention
            df_dsa = df_adj.loc[self.target_labels, self.target_labels]
            assert np.array_equal(df_dsa.index.to_numpy(), df_dsa.columns.to_numpy()) # self-attention: rows == cols
            assert np.array_equal(df_dsa.index.map(tv_map).to_numpy(), tv_order)      # rows == target variable sequential order
            
            # decoder cross-attention
            df_dca = df_adj.loc[self.target_labels, self.input_labels]
            assert np.array_equal(df_dca.index.map(tv_map).to_numpy(), tv_order)      # rows == target variable sequential order
            assert np.array_equal(df_dca.columns.map(iv_map).to_numpy(), iv_order)    # cols == input variable sequential order
        else:
            # Build the order for DAG adjacency matrix based on shared_embedding flag
            if shared_embedding:
                # Use shared ordering: source -> input -> target (sorted by shared ID)
                all_labels = self.source_labels + self.input_labels + self.target_labels
                df_adj = self.scm.adjacency(nodes=all_labels, positive_child=True, as_dataframe=True)
            else:
                df_adj = self.scm.adjacency(positive_child=True, as_dataframe=True)
            
            # Create attention masks for transformer
            # Rows are queries, columns are keys
            
            # Decoder 1 cross-attention: X attends to S (rows=input, cols=source)
            df_d1ca = df_adj.loc[self.input_labels, self.source_labels]
            assert np.array_equal(df_d1ca.index.map(iv_map).to_numpy(), iv_order)      # rows == input variable sequential order
            assert np.array_equal(df_d1ca.columns.map(sv_map).to_numpy(), sv_order)    # cols == source variable sequential order
            
            # Decoder 1 self-attention: X attends to X (rows=input, cols=input)
            df_d1sa = df_adj.loc[self.input_labels, self.input_labels]
            assert np.array_equal(df_d1sa.index.to_numpy(), df_d1sa.columns.to_numpy()) # self-attention: rows == cols
            assert np.array_equal(df_d1sa.index.map(iv_map).to_numpy(), iv_order)       # rows == input variable sequential order
            
            # Decoder 2 masks only if target_labels is non-empty
            df_d2ca = None
            df_d2sa = None
            if self.target_labels:
                # Decoder 2 cross-attention: Y attends to X (rows=target, cols=input)
                df_d2ca = df_adj.loc[self.target_labels, self.input_labels]
                assert np.array_equal(df_d2ca.index.map(tv_map).to_numpy(), tv_order)       # rows == target variable sequential order
                assert np.array_equal(df_d2ca.columns.map(iv_map).to_numpy(), iv_order)     # cols == input variable sequential order
                
                # Decoder 2 self-attention: Y attends to Y (rows=target, cols=target)
                df_d2sa = df_adj.loc[self.target_labels, self.target_labels]
                assert np.array_equal(df_d2sa.index.to_numpy(), df_d2sa.columns.to_numpy()) # self-attention: rows == cols
                assert np.array_equal(df_d2sa.index.map(tv_map).to_numpy(), tv_order)       # rows == target variable sequential order
        
        
        # ------------ get SCM graph visualization --------------------
        graph = self.scm.to_graphviz()
        
        
        # ---------------------- metadata -----------------------------
        if meta_dict is not None:
            for key, value in meta_dict.items():
                self.meta[key] = value 
        
        
        # ---------------------- export -------------------------------
        makedirs(save_dir, exist_ok=True)
        
        # Export data arrays - handle split vs no-split cases
        if split_info is not None:
            # Export separate train and test files
            # Train data
            if "y" in train_data and "s" in train_data:
                np.savez_compressed(join(save_dir, "ds_train.npz"), 
                                    x=train_data["x"], y=train_data["y"], s=train_data["s"])
            elif "s" in train_data:
                np.savez_compressed(join(save_dir, "ds_train.npz"), 
                                    x=train_data["x"], s=train_data["s"])
            elif "y" in train_data:
                np.savez_compressed(join(save_dir, "ds_train.npz"), 
                                    x=train_data["x"], y=train_data["y"])
            else:
                np.savez_compressed(join(save_dir, "ds_train.npz"), x=train_data["x"])
            
            # Test data
            if "y" in test_data and "s" in test_data:
                np.savez_compressed(join(save_dir, "ds_test.npz"), 
                                    x=test_data["x"], y=test_data["y"], s=test_data["s"])
            elif "s" in test_data:
                np.savez_compressed(join(save_dir, "ds_test.npz"), 
                                    x=test_data["x"], s=test_data["s"])
            elif "y" in test_data:
                np.savez_compressed(join(save_dir, "ds_test.npz"), 
                                    x=test_data["x"], y=test_data["y"])
            else:
                np.savez_compressed(join(save_dir, "ds_test.npz"), x=test_data["x"])
            
            # Export split info
            with open(join(save_dir, 'split_info.json'), 'w', encoding="utf-8") as file:
                json.dump(split_info, file, indent=2, sort_keys=True, ensure_ascii=False)
            print(f"Exported ds_train.npz and ds_test.npz with split_info.json")
            
        else:
            # Original behavior: single ds.npz file
            if source_np is not None:
                if target_np is not None:
                    np.savez_compressed(join(save_dir, "ds.npz"), x=input_np, y=target_np, s=source_np)
                else:
                    np.savez_compressed(join(save_dir, "ds.npz"), x=input_np, s=source_np)
            else:
                if target_np is not None:
                    np.savez_compressed(join(save_dir, "ds.npz"), x=input_np, y=target_np)
                else:
                    np.savez_compressed(join(save_dir, "ds.npz"), x=input_np)
        
        # Export attention masks based on source_labels presence
        if self.source_labels is None:
            # Export three separate attention mask files
            df_esa.to_csv(join(save_dir, "enc_sef_att_mask.csv"))
            df_dsa.to_csv(join(save_dir, "dec_self_att_mask.csv"))
            df_dca.to_csv(join(save_dir, "dec_cross_att_mask.csv"))
        else:
            # Export full DAG adjacency matrix
            df_adj.to_csv(join(save_dir, "dag_adj_mask.csv"))
            
            # Export attention masks for transformer (2 or 4 depending on target_labels)
            df_d1ca.to_csv(join(save_dir, "dec1_cross_att_mask.csv"))  # X attends to S
            df_d1sa.to_csv(join(save_dir, "dec1_self_att_mask.csv"))   # X attends to X
            
            if df_d2ca is not None and df_d2sa is not None:
                df_d2ca.to_csv(join(save_dir, "dec2_cross_att_mask.csv"))  # Y attends to X
                df_d2sa.to_csv(join(save_dir, "dec2_self_att_mask.csv"))   # Y attends to Y
                print("Exported 4 sliced attention masks: dec1_cross, dec1_self, dec2_cross, dec2_self")
            else:
                print("Exported 2 sliced attention masks: dec1_cross, dec1_self (no target variables)")
        
        # Export metadata
        with open(join(save_dir, 'meta.json'),'w', encoding="utf-8")  as file:
            json.dump(self.meta, file, indent=2, sort_keys=True, ensure_ascii=False)
        
        # Export variable mappings based on shared_embedding flag
        if shared_embedding and shared_vars_map is not None:
            # Export single shared variable mapping
            with open(join(save_dir, 'shared_vars_map.json'),'w', encoding="utf-8")  as file:
                json.dump(shared_vars_map, file, indent=2, sort_keys=True, ensure_ascii=False)
            print(f"Exported shared variable mapping with {len(shared_vars_map)} variables")
        else:
            # Export category-specific variable mappings (original behavior)
            # Export input mappings
            with open(join(save_dir, 'input_vars_map.json'),'w', encoding="utf-8")  as file:
                json.dump(iv_map, file, indent=2, sort_keys=True, ensure_ascii=False)
                
            with open(join(save_dir, 'input_feat_map.json'),'w', encoding="utf-8")  as file:
                json.dump(if_map, file, indent=2, sort_keys=True, ensure_ascii=False)
                
            # Export target mappings
            with open(join(save_dir, 'target_vars_map.json'),'w', encoding="utf-8")  as file:
                json.dump(tv_map, file, indent=2, sort_keys=True, ensure_ascii=False)
                
            with open(join(save_dir, 'target_feat_map.json'),'w', encoding="utf-8")  as file:
                json.dump(tf_map, file, indent=2, sort_keys=True, ensure_ascii=False)
            
            # Export source mappings if present
            if sv_map is not None:
                with open(join(save_dir, 'source_vars_map.json'),'w', encoding="utf-8")  as file:
                    json.dump(sv_map, file, indent=2, sort_keys=True, ensure_ascii=False)
                    
                with open(join(save_dir, 'source_feat_map.json'),'w', encoding="utf-8")  as file:
                    json.dump(sf_map, file, indent=2, sort_keys=True, ensure_ascii=False)
        
        # Export normalization stats if normalization was applied
        if norm_stats:
            with open(join(save_dir, 'normalization.json'),'w', encoding="utf-8")  as file:
                json.dump(norm_stats, file, indent=2, sort_keys=True, ensure_ascii=False)

        # Export dataset metadata for evaluation functions (NEW)
        dataset_metadata = self._generate_dataset_metadata(shared_vars_map)
        with open(join(save_dir, 'dataset_metadata.json'),'w', encoding="utf-8") as file:
            json.dump(dataset_metadata, file, indent=2, sort_keys=True, ensure_ascii=False)
        print(f"Exported dataset_metadata.json with causal structure and variable info")

        # Export ATE ground truth for intervention evaluation (Monte Carlo only).
        # ``compute_ate=False`` skips it entirely: randomly sampled DAGs (DAG sweeps)
        # are only scored on structure recovery (SHD), so the two 50k-sample Monte
        # Carlo passes would be pure overhead.
        if not compute_ate:
            print("Skipping ATE ground truth export (compute_ate=False)")
        elif self.source_labels:
            try:

                # Uses default PAPER_INTERVENTIONS from compute_ate_ground_truth()
                # Unified scheme: ID=±0.5, OOD=1.5 (see PAPER_INTERVENTIONS)
                
                # Compute Monte Carlo ground truth (used for both ground truth and model evaluation)
                ate_ground_truth = self.compute_ate_ground_truth(
                    do_values=None,  # Uses PAPER_INTERVENTIONS by default
                    n_samples=50000,
                    seed=seed
                )
                
                # Package with metadata for evaluation functions
                # Monte Carlo only - analytical removed due to incorrect E[f(eps)] = f(0) assumption
                #
                # The descriptors below mirror PAPER_INTERVENTIONS, whose roles are
                # defined on the 5-source paper SCMs.  They MUST be filtered to the
                # sources this SCM actually has: randomly sampled DAGs (DAG sweeps)
                # may have only S1..S2, and compute_ate_ground_truth() already skips
                # the missing ones.  Exporting the unfiltered list made downstream
                # evaluation call do(S3=...) and raise
                # "Intervention targets unknown node 'S3'".
                intervention_specs = {
                    "S1": {"values": [0.5], "type": "in_distribution", "role": "negative_control"},
                    "S2": {"values": [0.5, 1.5], "type": "mixed", "role": "positive_control", "ood_values": [1.5]},
                    "S3": {"values": [-0.5, 1.5], "type": "mixed", "role": "structure_test", "ood_values": [1.5]},
                    "S5": {"values": [-0.5, 1.5], "type": "mixed", "role": "confounding_test", "ood_values": [1.5]},
                }
                existing_sources = set(self.source_labels) & set(self.scm.specs)
                ate_export = {
                    "description": "Ground-truth ATE = E[X|do(S=s)] - E[X|do(S=0)] via Monte Carlo sampling",
                    "interventions": {
                        node: spec for node, spec in intervention_specs.items()
                        if node in existing_sources
                    },
                    "computation_method": "monte_carlo",
                    "n_samples": 50000,
                    "monte_carlo": {
                        "ate": ate_ground_truth["ate"],
                        "baseline": ate_ground_truth["baseline"],
                        "treated": ate_ground_truth["treated"],
                    },
                }
                
                with open(join(save_dir, 'ate_ground_truth.json'), 'w', encoding="utf-8") as file:
                    json.dump(ate_export, file, indent=2, sort_keys=True, ensure_ascii=False)
                print(f"Exported ate_ground_truth.json (Monte Carlo, n=50000) with {len(ate_ground_truth['ate'])} interventions")
                
            except Exception as e:
                print(f"Warning: Could not compute ATE ground truth: {e}")
        else:
            print("Skipping ATE ground truth export (no source_labels defined)")

        # ---- render DAG pdf (optional: needs the Graphviz 'dot' binary) ----
        # On some environments (e.g. HPC clusters) the system binary is not
        # available. The pdf is purely informative, so skip it instead of
        # failing the whole dataset generation.
        try:
            graph.render(str(join(save_dir, 'graph')), format="pdf", cleanup=True)
        except Exception as e:
            print(f"Warning: skipping DAG pdf rendering ('dot' unavailable?): {e}")


# --------------------------- Example -------------------------------- #
if __name__ == "__main__":
    # Authoring: symbolic node specs using parent names and eps_<name> for noise.
    specs = [
        NodeSpec("X", [],            "eps_X"),                  # exogenous
        NodeSpec("Z", [],            "eps_Z"),                  # exogenous (scaled)
        NodeSpec("Y", ["X", "Z"],    "X + w_YZ*Z + eps_Y"),     # structural
    ]

    # Parameters
    params = {"w_YZ":0.01}
    
    # Noise vars
    
    singles = {"X": lambda rng,n: rng.standard_normal(n)} # independent
    
    group = GroupNoise(
            nodes=("Z","Y"),
            sampler=lambda rng,n: rng.multivariate_normal(
                mean=[10,0],
                cov=[
                    [1.0, 0.1],
                    [0.1, 1.0]
                    ],
                size=n
                )
            )
    
    nm = NoiseModel(singles=singles, groups=[group])
    
    
    # ------------------- Quick Test Specs class -----------------------
    
    specs_example = Spec(
        name= "example",
        nodes=specs,
        params=params,
        noise_model = nm,
        )
    
    specs_example.validate()
    scm = specs_example.to_scm()
    
    # -------------------- Quick Test SCM class ------------------------
    
    # Observational sampling
    df_obs = scm.sample(n=5, seed=42)

    # Interventional sampling: do(Z = 1.0)
    scm_do = scm.do({"Z": 1.0})
    df_do = scm_do.sample(n=5, seed=42)

    print(df_obs.head())
    print(df_do.head())
    
    
    # Build adjacency
    df_adj = scm.adjacency()
    print(df_adj)
