# Structure-Learning Benchmarks (NOTEARS, DAGMA, PC)

External baselines for DAG recovery, wired into the existing causaliT evaluation
plumbing so their numbers are directly comparable to the models'.

Code: `causaliT/benchmarks/`
Tests: `tests/test_benchmarks.py`

---

## 1. What is implemented

| Method           | Class                    | Source of the code                                | Extra dependency |
|------------------|--------------------------|---------------------------------------------------|------------------|
| `notears_linear` | linear SEM, continuous   | vendored paper code (`vendor/notears/`)           | none (scipy)     |
| `notears_mlp`    | nonlinear, per-node MLP  | vendored paper code (`vendor/notears/`)           | `torch`          |
| `dagma_linear`   | linear SEM, continuous   | `dagma` package (authors' own release)            | `dagma`          |
| `dagma_mlp`      | nonlinear, per-node MLP  | `dagma` package                                   | `dagma`          |
| `pc`             | constraint-based, CPDAG  | `causal-learn` package (CMU, reference impl.)     | `causal-learn`   |

Nothing is reimplemented from scratch. DAGMA and PC come from the authors'
published packages; NOTEARS has no PyPI release, so `notears_linear.py` and
`notears_mlp.py` from https://github.com/xunzheng/notears are vendored verbatim
under `causaliT/benchmarks/vendor/notears/` together with their Apache-2.0
licence and a `PROVENANCE.md` recording the commit. The wrappers in
`causaliT/benchmarks/methods/` never modify the algorithms - they only convert
arguments and results.

Install everything with:

```bash
pip install -r requirements.txt   # includes dagma and causal-learn
```

Check what is currently runnable:

```bash
python -m causaliT.benchmarks.cli list
```

---

## 2. Layout

```
causaliT/benchmarks/
  data.py          ds.npz (token tensors) -> (n_samples, N) design matrix
  base.py          BenchmarkResult, method registry, paper defaults
  methods/         one thin wrapper per method (lazy third-party imports)
  postprocess.py   estimated W -> canonical causaliT DAG blocks
  runner.py        fit + write the standard eval artefacts
  cli.py           `list` / `run` commands
  vendor/notears/  verbatim paper code + LICENSE + PROVENANCE.md
```

The evaluation side gained one shared module,
`causaliT/evaluation/eval_funs/helpers/eval_dag_report.py`, which holds the
model-free half of `eval_attention_scores` (`resolve_dag_dims`,
`write_dag_report`). Models and benchmarks both call it, so `dag_metrics.json`
and `learned_dag_edges.json` are produced by identical code:

```
model:     checkpoint -> attention -> query_dag_blocks -\
                                                         >-- write_dag_report
benchmark: design matrix -> W -> adjacency_to_blocks ----/
```

---

## 3. The two conventions that must not slip

**Orientation.** The papers write `X = X W + noise`, so `W[i, j] != 0` means
`i -> j` (rows are parents). causaliT's DAG masks are the opposite: rows are
children, columns parents (`dec_cross` is `(L_X, L_S)`). The conversion is a
single transpose in `postprocess.to_canonical_adjacency` and happens nowhere
else. A missing transpose would still yield plausible SHD numbers while scoring
the reverse graph, so `tests/test_benchmarks.py` pushes the ground-truth `W`
through the full pipeline and asserts it reproduces the true masks exactly.

**Column order.** `data.py` concatenates the token tensors sources-first
(`[S1..S_LS, X1..X_LX]`), which is the ordering `query_dag_blocks` assumes when
it slices a square `(N, N)` matrix into `cross` and `self`. The labels travel
with the matrix and are recorded in `benchmark_run.json`.

**Scores.** `write_dag_report` expects edge probabilities in `[0, 1]`:

- weighted methods (NOTEARS, DAGMA): `|W| >= w_threshold` -> `1.0`, else `0.0`
  (`score_mode: binary`, the papers' own reporting), or `|W| / max|W|`
  (`score_mode: scaled`) to keep a notion of confidence;
- PC returns a CPDAG: oriented edges get `1.0`, unoriented edges `0.5` in both
  directions. At the default `dag_threshold = 0.5` an unoriented edge counts as
  present both ways, which is the honest reading of an equivalence class (and
  makes `is_dag` report `False` for PC by construction).

---

## 4. Protocol decisions

**Fixed paper hyperparameters, no tuning.** Each method module exposes
`DEFAULT_PARAMS` copied from the paper/package defaults; sweeps never search over
them. `benchmark.params.<method>` can override any value, but doing so must be
reported.

**Why the nonlinear MLP width is not swept.** Both nonlinear variants are
per-node MLPs with `dims = [d, H, 1]`, and both papers use `H = 10` for every
graph size they report. The architecture is therefore size-independent by
construction: there is nothing to derive from `N` and no capacity search to run.
Fixing `H = 10` reproduces the published setting instead of inventing a new one.

**Seeds play the role of folds.** causaliT reports best/mean/worst across `k`
folds. The benchmarks have no folds, so each method is refitted once per seed and
stored under `seed_<i>`, which `write_dag_report` consumes exactly like
`fold_<i>`. Linear NOTEARS/DAGMA/PC are deterministic given the data, so their
spread is zero unless the data changes; the MLP variants vary through
initialisation. The reported spread is thus each method's true variability.

**Standardisation is on by default - and it matters.** `benchmark.standardize:
true` z-scores every column before fitting. Simulated linear SEMs are
*varsortable*: marginal variance grows along the topological order, and the
continuous methods can read the order off the scale rather than off the
dependence structure (Reisach et al., 2021, "Beware of the Simulated DAG"). On
this project's own synthetic chain, `notears_linear` recovers the DAG exactly on
raw data and clearly worse after standardising - the difference is the shortcut,
not the signal. `tests/test_benchmarks.py::TestMethods::
test_standardization_removes_the_varsortability_shortcut` pins this so a weak
benchmark score can never be "fixed" by silently dropping standardisation. If you
report raw-scale numbers, say so.

**Background knowledge is off by default.** causaliT's source variables are
exogenous, so any edge into an `S` node is necessarily wrong. The benchmarks do
not know this; `benchmark.forbid_into_sources: true` enforces it. It makes the
comparison *more* favourable to the baselines, so it is opt-in and must be
reported when used.

---

## 5. Usage

### 5.1 Standalone CLI

```bash
# List methods, availability and the paper defaults that will be used
python -m causaliT.benchmarks.cli list

# Fit on the dataset of an existing experiment folder (config only, no checkpoint)
python -m causaliT.benchmarks.cli run \
    --experiment experiments/.../my_run \
    --methods notears_linear,dagma_linear,pc \
    --seeds 0,1,2 \
    --csv_out benchmarks.csv
```

Artefacts land in `<experiment>/eval/eval_benchmark_<method>/files/`:

| File                     | Content                                                       |
|--------------------------|---------------------------------------------------------------|
| `dag_metrics.json`       | soft Hamming, `standard_shd_*`, zeroness, MEC - model schema   |
| `learned_dag_edges.json` | per-seed blocks + true mask + variable labels                  |
| `attention_labels.json`  | block descriptions                                             |
| `benchmark_run.json`     | raw `W` per seed, params, timings, package versions            |

`benchmark_run.json` stores the unthresholded `W`, so any `w_threshold` or
`score_mode` can be re-scored offline without refitting.

### 5.2 Config section

```yaml
benchmark:
  methods: [notears_linear, notears_mlp, dagma_linear, dagma_mlp, pc]
  seeds: [0, 1, 2]
  split: train
  standardize: true          # see the varsortability note above
  w_threshold: 0.3           # papers' pruning threshold on |W|
  score_mode: binary         # or "scaled"
  forbid_into_sources: false # background knowledge, opt-in
  params:                    # optional per-method overrides
    notears_linear: {lambda1: 0.05}
```

Defaults live in `runner.DEFAULT_BENCHMARK_CONFIG`; the config section overrides
them, and CLI/sweep arguments override the config.

### 5.3 Inside a DAG sweep

Set the trainer to `benchmark` and the sweep runs the baselines on exactly the
same generated datasets as the models, with the same folder structure:

```yaml
training:
  trainer: benchmark
```

`benchmark_function_for_sweep` (in `causaliT/euler_sweep/euler_sweep/cli.py`)
has the same signature as `train_function_for_sweep`, trains nothing, and returns
one row per method (`shd_cross_mean`, `soft_hamming_cross_mean`,
`mec_distance_mean`, `seconds_mean`, ...), which it also writes to
`<run>/benchmark_summary.csv` since the sweep discards trainer return values.
See `DAGSWEEP_OPTUNA.md`.

Two ready-to-run smoke tests double as annotated templates (each ~10 s, CPU only):

| Folder | Covers |
| --- | --- |
| `experiments/0_TESTS/FUN_benchmarks/` | score-based path (`notears_linear` active, DAGMA / MLP arms commented) |
| `experiments/0_TESTS/FUN_benchmarks_pc/` | constraint-based path: CPDAG scores, MEC metrics, the load-bearing `dag_threshold` |

```bash
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep --exp_id 0_TESTS/FUN_benchmarks
```

Both share the DAG recipe of `0_TESTS/FUN_dagsweep`, so their datasets are
byte-identical to the model smoke test's and the numbers are comparable. Their
configs explain, key by key, why a benchmark folder needs no `model` /
`size_derived` / `model_seeds` block and why Optuna stays off.


### 5.4 From Python

```python
from causaliT.benchmarks.runner import run_benchmarks, summarize_benchmarks

results = run_benchmarks("experiments/.../my_run", methods=["pc"])
rows = summarize_benchmarks(results)
```

---

## 6. Adding a method

1. Add `causaliT/benchmarks/methods/<name>.py` with `DEFAULT_PARAMS` and
   `fit(X, **params) -> BenchmarkResult`, returning `W` in **paper orientation**
   and importing the third-party package *inside* `fit`.
2. Register it in `base.METHOD_SPECS`, `METHOD_DESCRIPTIONS` and
   `METHOD_REQUIREMENTS`.
3. If it is not on PyPI, vendor the paper code under `vendor/<name>/` with its
   licence and a `PROVENANCE.md`.
4. Add it to the parametrised contract tests in `tests/test_benchmarks.py`.

Nothing else changes: the registry, runner, CLI and sweep integration pick it up.

---

## 7. References

- Zheng et al., *DAGs with NO TEARS: Continuous Optimization for Structure
  Learning*, NeurIPS 2018 - https://github.com/xunzheng/notears
- Zheng et al., *Learning Sparse Nonparametric DAGs*, AISTATS 2020 (NOTEARS-MLP)
- Bello et al., *DAGMA: Learning DAGs via M-matrices and a Log-Determinant
  Acyclicity Characterization*, NeurIPS 2022 - https://github.com/kevinsbello/dagma
- Spirtes & Glymour, *An Algorithm for Fast Recovery of Sparse Causal Graphs*,
  1991 (PC) - implementation: https://github.com/py-why/causal-learn
- Reisach et al., *Beware of the Simulated DAG!*, NeurIPS 2021 (varsortability)
