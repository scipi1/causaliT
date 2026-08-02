# euler_optuna — Optuna Capacity Search for causaliT

Hyperparameter optimisation module for causaliT models using Optuna.

See **[QUICK_START.md](QUICK_START.md)** for full usage instructions, design rationale,
and FAQ on protocol choices.

---

## Purpose

Before causal training, each (model × dataset) combination needs a **capacity search** to
ensure the architecture has enough expressivity to reconstruct the dataset well. Without
this step, differences in ATE or structural-learning performance between architectures
might simply reflect capacity differences, not causal inductive-bias differences.

The search optimises `val_x_mae` (mean absolute error on the reconstructed variable `X`)
using the **best checkpoint** from a short training run. All causal penalties are
disabled so only reconstruction quality drives the search.

---

## Structure

```
euler_optuna/
├── QUICK_START.md               # Usage guide + design decisions
└── euler_optuna/                # Inner package
    ├── optuna_opt.py            # Generic OptunaStudy class (framework core)
    ├── optuna_parallel.py       # SLURM parallel execution
    ├── cli.py                   # causaliT-specific CLI + sampling + protocol
    └── optuna_worker.py         # SLURM array task worker
```

### Key components

**`cli.py`** — The main entry point. Contains:
- `paramsopt` CLI command (create / resume / summary modes)
- `_build_optuna_protocol()` — per-model config overrides that freeze structural
  learning and enable early stopping for each search trial
- `_sample_hyperparams()` — model-specific parameter sampling from `BASELINE_SAMPLING_BOUNDS`
- `train_function_for_optuna()` — wraps `trainer()` and extracts the optimisation metric

**`optuna_opt.py`** — Framework-agnostic `OptunaStudy` class. Handles study creation,
SQLite storage, sampler/pruner configuration, and trial iteration.

**`optuna_parallel.py`** — SLURM job-array submission. Generates one job per trial so
each trial runs on an independent GPU (scheduler-friendly).

**`optuna_worker.py`** — Entry point for individual SLURM array tasks.

---

## Optimisation Protocol

Every trial applies the following overrides automatically (see `_build_optuna_protocol()`):

| Override | Value | Reason |
|----------|-------|--------|
| `lambda_l1` / `lambda_hsic` / `lambda_notears` / `lambda_group_l1` | 0 | Disable causal penalties |
| `hard_mask_files` | `[]` | Disable gradient routing |
| `k_fold` | 1 | Single 80/20 split (speed) |
| `early_stopping` | patience=10, monitor=`val_x_mae` | Stop when reconstruction plateaus |
| `best=True` | — | Use best-checkpoint metrics, not final epoch |

The protocol is saved as `optuna/optuna_protocol.json` alongside `study.db` for
reproducibility.

---

## Models supported

- `proT`
- `StageCausaliT`
- `SingleCausalLayer` / `SingleCausalLayerRes`
- `NoiseAwareSingleCausalLayer` / `NoiseAwareSingleCausalLayerRes`

---

## Attention bug fix

Running Optuna with `n_heads > 1` revealed a pre-existing bug in
`causaliT/core/modules/attention.py`: the SVFA architecture uses separate head counts
for Q/K (structure) and V (value). When `n_heads_struct=1` (3-D Q/K) but
`n_heads_value > 1` (4-D V), `ScaledDotSoftmax` would fail with a shape mismatch.

Fixed by adding a mixed-head einsum branch:
```python
elif value.dim() == 4:
    # Q/K single-head (3D), V multi-head (4D): broadcast attention across V heads
    V = torch.einsum("bls,bshd->blhd", A, value)
```
The same fix was applied to `CausalCrossAttention`, `SigmoidCrossAttention`, and
`ToeplitzAttention`.
