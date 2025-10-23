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
import datetime
from os import makedirs
from os.path import join
import numpy as np
import pandas as pd
import sympy as sp
import json
from graphviz import Digraph


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
    ) -> None:
        
        self.noise_model = NoiseModel(singles=singles, groups=groups)
        specs_scm = Spec(
            name=name,
            nodes=specs,
            params=params,
            noise_model=self.noise_model,
        )
        self.scm = specs_scm.to_scm()
        self.input_labels = input_labels
        self.target_labels = target_labels
        
        # meta
        created = datetime.date.today().isoformat()
        self.meta: Dict[str, Any] = {
            "name": name,
            "created": created,
            "description": description,
        }
        
        
    def sample(self, n, seed=42):
        return self.scm.sample(n, seed)
    
    def get_numpy(self, mode, n, seed=42):
        # reshape the dataset into B x L x D, where
                # - B: batch/sample size
                # - L: is the sequence length
                # - D: is the feature length (value and variable)
        
        df = self.sample(n,seed)
        
        if mode == "flat":
            
            def to_numpy_(label):
                
                df_ = pd.melt(df[label], id_vars=None, ignore_index=False)
                unique_vars = df_["variable"].unique()
                
                # convert vars --> int and save map
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
            
            input_np, iv_map, if_map, iv_order = to_numpy_(self.input_labels)
            target_np, tv_map, tf_map, tv_order = to_numpy_(self.target_labels)
            
            return input_np, (iv_map, if_map, iv_order), target_np , (tv_map, tf_map, tv_order)

            
    def generate_ds(self, mode, n, save_dir: Union[str, Path]=None, meta_dict: dict=None, seed=42):
        
        # get numpy array
        input_np, (iv_map, if_map, iv_order), target_np , (tv_map, tf_map, tv_order) = self.get_numpy(mode, n, seed)
        print("numpy arrays generated")
        
        # todo train/test split
        
        
        # ------------------ make attention masks -----------------------
        # rows for queries
        # cols for keys
        
        df_adj = self.scm.adjacency(positive_child= True, as_dataframe=True)
        
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
        
        
        # ------------ get SCM graph visualization --------------------
        graph = self.scm.to_graphviz()
        
        
        # ---------------------- metadata -----------------------------
        if meta_dict is not None:
            for key, value in meta_dict.items():
                self.meta[key] = value 
        
        
        # ---------------------- export -------------------------------
        makedirs(save_dir, exist_ok=True)
        np.savez_compressed(join(save_dir, "ds.npz"), x=input_np, y=target_np)
        
        df_esa.to_csv(join(save_dir, "enc_sef_att_mask.csv"))
        df_dsa.to_csv(join(save_dir, "dec_self_att_mask.csv"))
        df_dca.to_csv(join(save_dir, "dec_cross_att_mask.csv"))
        
        with open(join(save_dir, 'meta.json'),'w', encoding="utf-8")  as file:
            json.dump(self.meta, file, indent=2, sort_keys=True, ensure_ascii=False)
            
        with open(join(save_dir, 'input_vars_map.json'),'w', encoding="utf-8")  as file:
            json.dump(iv_map, file, indent=2, sort_keys=True, ensure_ascii=False)
            
        with open(join(save_dir, 'input_feat_map.json'),'w', encoding="utf-8")  as file:
            json.dump(if_map, file, indent=2, sort_keys=True, ensure_ascii=False)
            
        with open(join(save_dir, 'target_vars_map.json'),'w', encoding="utf-8")  as file:
            json.dump(tv_map, file, indent=2, sort_keys=True, ensure_ascii=False)
            
        with open(join(save_dir, 'target_feat_map.json'),'w', encoding="utf-8")  as file:
            json.dump(tf_map, file, indent=2, sort_keys=True, ensure_ascii=False)

        graph.render(str(join(save_dir, 'graph')), format="pdf", cleanup=True)


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
