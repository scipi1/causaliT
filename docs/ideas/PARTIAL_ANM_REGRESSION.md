### Partial Additive Noise Models (ANM) Regression

#### Objective
We observed empirically that just minimizing independence doesn't make the transformer learn the true causal structure. The objective of this approach is to make the transformer training closer to an Additive Noise Model (ANM) regression procedure or, at least, understand the limits.

#### Notation

$X$, $S$ are vectors. $X_j$, $S_k$ are variables, or nodes. Candidate parents of $X_j$ can come from cross-attention ($S_i\to X_j$) or self-attention ($X_k\to X_j$).


#### ANM
In bivariate ANM (Peters - 2014), we assume $X_j=f(S_k)+\epsilon_{X_j}$ with $\epsilon_{X_j}\perp S_k$ in the causal direction.
1) Regress $X_j$ on $S_k$ 
2) Calculate the residual $\epsilon_{X_j}=X_j-f(S_k)$ 
3) check the independence with the Hilbert-Schmidt independence criterion $\operatorname{HSIC}(\epsilon_{X_j}, S_k)$.
    - If the model class is correct and the right parent set is used, the residual should be independent of the parents.
    - Residual independence alone is **not** a proof that $S_k$ is a direct parent: $S_k$ could be independent of $X_j$ or a proxy for a true parent.
    - Residual dependence means that the current regression/parent set has not explained all dependence, but it does not by itself prove that $S_k$ is not a parent.

NOTE: for multivariate ANM, like our case, the whole sufficient parent set is included in the independence condition: $X_j=f_j(\operatorname{PA}(X_j))+\epsilon_j$ with $\epsilon_j\perp \operatorname{PA}(X_j)$. Our method should therefore be seen as ANM-inspired regularization, not an exact ANM identifiability test at every stochastic step.


The problem with transformers is that they can perform the regression using a dense mixture of variables. Batch-consistent key dropout aims at reducing the number of variables available in a batch, creating stochastic partial regressions closer to the ANM setting.

#### Batch Consistent Key Dropout workflow
The idea is to sample, for each step, a subset of the key vector and reduce the order of the regression to be in a more similar condition to ANM.
Another advantage (maybe) is that the reconstruction model $f$ learns to predict from different and limited inputs and the hope is that 

- At each step, a mask is sampled with $\operatorname{Bern}(p)$ and subsets of keys are selected: $\mathcal{K}^{(t)}_S\subset S$ for cross-attention and $\mathcal{K}^{(t)}_X\subset X$ for self-attention.
- Perform partial regression $\hat X_j=f_j(\mathcal{K}^{(t)}_S,\mathcal{K}^{(t)}_X)$.
- Calculate reconstruction loss to improve regressor
- Calculate *valid* residuals $\epsilon_{X_j}=X_j-\hat X_j$
    - NOTE: the residual will be high in early training due to the poor fit of the regressor $f$.
    - NOTE: *valid* residual means residuals of variables that have been regressed on at least one active key.
    - Cross-HSIC uses residuals of $X_j$ that had at least one active $S$ key.
    - Self-HSIC uses residuals of $X_j$ that had at least one active $X$ key.
    - Reconstruction uses the union of cross-active and self-active $X_j$ variables.
- Calculate structure loss as averaged or per-variable independence with HSIC, e.g. $\frac{1}{|\mathcal{K}^{(t)}_S|}\sum_{S_i\in\mathcal{K}^{(t)}_S}\operatorname{HSIC}(\epsilon_{X_j},S_i)$ for cross-attention, and analogously for self-attention.
- Update structural and reconstruction parameters


#### Potential failures (Not yet addressed or confirmed)
1) When we sample a structure with active path $S_f\to X_j$ with $S_f\notin\operatorname{PA}(X_j)$ in the real SCM, the $\operatorname{HSIC}\gg 0$. Since we ask to minimize the HSIC, does it lead to weaken the wrong edge $S_f\to X_j$? In a more formal statement, consider the adjaciency matrix $A$ and the element $A_{jf}$ representing the edge $S_f\to X_j$, do we have that $\arg_{\theta_S} \min \operatorname{HSIC}(\epsilon_{X_j}, S_f)\Rightarrow A_{jf}=0$? Or the best way to minimize the independence is to increase the $A_{jf}$? This might need a mathematical proof and heuristic validation. Or the only way to turn off an edge is with sparsity regularization?

2) In early training, the regressor is poor and the information we get from HSIC might be ill-posing the structure optimization. We could include some warmup epochs where only the reconstruction is optimized. Since Sigmoid cross and Toeplitz self (with zero bias) initialize $A$ to dense matrix, $f$ can learn to regress well from any subset of $S$.

3) Omitted-parent bias: if dropout removes a true parent, residuals may remain dependent on the kept variables even if those kept variables are also true parents. This can make HSIC penalize useful edges in partial regressions.

4) Proxy variables: a non-parent correlated with a true parent can reduce reconstruction loss and residual dependence. HSIC plus reconstruction may therefore select predictive proxies unless sparsity and architectural constraints are strong enough.

5) Attention is only a structural proxy. Value projections, embeddings, residual paths, and FFNs can move information in ways that make attention weights not perfectly causal.

6) Self-attention is harder than cross-attention: $S\to X$ is bipartite and acyclic by construction, while $X\to X$ needs self-loop removal, directionality, and acyclicity constraints.

7) HSIC estimator issues: finite batch size, kernel bandwidth, residual scale, and mixed discrete/continuous variables can strongly affect the signal. Adaptive or normalized HSIC may help but can also change gradient scale.

8) Dropout schedule matters: too much dropout rarely samples sufficient parent sets; too little dropout returns to dense transformer regression. Cross- and self-attention may need different schedules.

9) Averaged HSIC can dilute edge-specific gradients. Per-variable or attention-weighted HSIC may be more informative, but attention-weighted HSIC can also kill discovery if low weights suppress gradients too early.


## Experiments

### 1) Subsequent Structure-Reconstruct

Train in alternating reconstruction/structure phases while keeping key-dropout ON. Here "dense" means dense initialized attention over the sampled keys, not full-vector regression: the predictor should learn $f: \mathcal{K}\to X$ under the partial-key distribution.

Proposed schedule:

1. **Reconstruction phase:** freeze structure, train only reconstruction parameters with $p=p_\text{start}$ and HSIC off. This should make residuals meaningful before using them for structure learning.
2. **Structure phase:** freeze reconstruction, train only structural parameters with HSIC + sparsity/acyclicity. Gradient routing already separates objectives; alternation additionally makes the residual-generating function stationary during structure updates.
3. Decrease $p$ to include larger parent subsets and repeat reconstruction/structure phases.
4. Continue until low $p$ or $p=0$, optionally followed by a short low-LR joint fine-tuning phase.

Sub-experiments:

**H1 — Does reconstruction warmup make HSIC meaningful?**

- **Hypothesis:** HSIC is informative only after the regressor has learned a reasonable partial predictor.
- **Experiment:** train only reconstruction parameters with key-dropout ON and HSIC off. Then freeze the model and measure residual-HSIC for true parents, independent non-parents, proxy non-parents, and omitted-parent cases.
- **Expected:** residual-HSIC should be lower for sufficiently explained true-parent mechanisms; independent non-parents should be trivially low; proxy and omitted-parent cases may remain ambiguous/high.
- **Diagnostic:** edge-class residual-HSIC before any structure optimization.

**H2 — Does alternating help beyond gradient routing?**

- **Hypothesis:** even with gradient routing, structure learning is cleaner when the residual-generating function $f$ is frozen during structure updates.
- **Experiment:** compare joint gradient-routed training against alternating reconstruction/structure phases at fixed $p$.
- **Expected:** if residual stationarity matters, alternating should improve edge separation relative to joint training.
- **Diagnostic:** during structure-only phases, track whether $\operatorname{score}(\text{true edges})-\operatorname{score}(\text{false edges})$ increases more than in joint training.

**H3 — Does the dropout curriculum help?**

- **Hypothesis:** high-to-low key dropout first enforces partial regressions, then gradually exposes more complete parent sets needed for multivariate ANM.
- **Experiment:** compare alternating at fixed $p$ against alternating with decreasing $p$.
- **Expected:** the curriculum should reduce omitted-parent bias and improve final graph recovery if complete parent sets become necessary later.
- **Diagnostic:** edge precision/recall/SHD and true-vs-false edge score margin across curriculum stages.

**H4 — Full method vs current baseline**

- **Hypothesis:** the combination of reconstruction warmup, residual-stationary structure phases, and dropout curriculum improves causal structure learning.
- **Experiment:** compare current joint training + gradient routing + key dropout against the full subsequent structure-reconstruct schedule.
- **Expected:** the full method should improve graph metrics without degrading reconstruction too much.
- **Diagnostic:** final SHD/F1/AUC, reconstruction error, cross/self HSIC, sparsity, and edge-class trajectories.

Important caveat: decreasing $p$ does not guarantee that "relevant edges win" by itself. Wrong but predictive proxy edges may survive unless sparsity, acyclicity, and the independence signal jointly make true edges preferable.

## 2) Annealed Dense-to-Sparse Attention Bias

Motivation: HSIC gradients already indicate whether an edge affects residual independence, but if an initialized edge has near-zero gradient it may simply stay non-zero. We therefore add a global sparsifying prior by drifting the attention bias from dense to sparse:

$$
A_{ij}=\sigma(s_{ij}+b(t)), \qquad b(0)\approx 0,\quad b(T)\approx -20.
$$

Low-gradient edges then disappear by default, while useful edges must develop sufficiently positive scores $s_{ij}$ to survive.

**H5 — Does bias drift prune inactive/spurious edges?**

- **Hypothesis:** an increasingly negative attention bias turns sparsity into the default, so only edges supported by reconstruction/HSIC gradients remain active.
- **Experiment:** compare no bias drift, bias drift from epoch 0, bias drift after reconstruction warmup, and bias drift combined with the alternating schedule.
- **Expected:** false/proxy edges decay as $b(t)$ decreases; true edges survive by increasing their logits/scores before saturation.
- **Diagnostic:** track $\operatorname{score}(\text{true edges})-\operatorname{score}(\text{false edges})$ and $A(\text{true edges})-A(\text{false edges})$ during the bias schedule.

Implementation note: Toeplitz self-attention already has a `gate_bias`. Sigmoid cross-attention would need an explicit additive bias, e.g. $A=\sigma(\text{scores}/\tau+b_{cross})$. The drift should probably start after reconstruction warmup to avoid killing true edges before useful gradients emerge.

---

### Implementation

#### New components

**`causaliT/training/anm_staged_trainer.py`** — main orchestrator.

- `anm_alternating_trainer(config, data_dir, save_dir, ...)` runs an arbitrary sequence of stages defined in `config['anm_training']['stages']`.
- Data splits are computed **once** and shared across all stages (consistent train/val).
- Checkpoints are **chained**: each stage warm-starts from the last `.ckpt` of the previous.
- `StageEvalCallback` captures DAG metrics and `score(true edges) − score(false edges)` at stage end and every N epochs within a stage (H1/H2/H3 diagnostics).

Per-stage config keys (flat dict, overlaid on base config):

| Key | Effect |
|---|---|
| `name` | Stage directory name |
| `max_epochs` | Epochs for this stage |
| `lambda_hsic_cross`, `lambda_hsic_self` | HSIC weights |
| `lambda_recon` | Reconstruction weight (0 = structure-only stage) |
| `freeze_structural_params` | `requires_grad_(False)` on θ_S (needs `use_gradient_routing=True`; else falls back to `lambda_hsic=0`) |
| `freeze_reconstruction_params` | `requires_grad_(False)` on θ_R (needs `use_gradient_routing=True`; else falls back to `lambda_recon=0`) |
| `batch_key_dropout_p` | Fixed BKD drop probability for this stage (no within-stage annealing) |
| `use_gate_bias_annealing`, `gate_bias_start`, `gate_bias_end`, `gate_bias_anneal_epochs` | H5 bias drift |
| `eval_every_n_epochs`, `eval_dag` | Mid-stage *inline* snapshot frequency (cheap, in-process, no checkpoint load) |
| `evaluation.functions` | List of post-stage eval functions to run after `train_single_fold` returns (see below) |

#### Per-stage post-training evaluation (`evaluation.functions`)

After each stage's training completes, the orchestrator checks whether the stage spec has an `evaluation: {functions: [...]}` block and, if so, calls `run_evaluations_from_config(stage_dir, functions=[...])`.  Errors in evaluation are non-fatal and only produce a warning.

Available function names (same registry as the standard trainer):

| Function name | What it does |
|---|---|
| `eval_anm_residual_hsic` | **H1 diagnostic** — loads the stage checkpoint in eval mode (BKD off → dense attention), runs the validation split through the model, computes `HSIC(ε_j, S_i)` for every (X_j, S_i) pair, classifies pairs as `true_parent / false_parent / independent` against the true DAG, saves `edge_hsic.csv` + `summary.json` + heatmap |
| `eval_train_metrics` | Training loss/metric curves |
| `eval_attention_scores` | DAG recovery metrics from final checkpoint (soft Hamming, MEC) |
| `eval_attention_evolution` | Attention evolution across checkpoints (slow) |
| `eval_interventions` | ATE causal intervention tests |
| `fix_kfold_summary` | Fix tensor strings in `kfold_summary.json` |
| `enrich_kfold_summary` | Add aggregated statistics to `kfold_summary.json` |

**Typical usage pattern:** early stages request only `eval_anm_residual_hsic` (fast H1 diagnostic); the final joint fine-tuning stage additionally requests the classical evaluation functions.

**`SigmoidCrossAttention.gate_bias`** (plain `float`, default `0.0`) — additive bias in `σ(scores/τ + gate_bias)`. Mirrors the existing `ToeplitzAttention.gate_bias` (`nn.Parameter`) so the H5 annealer can handle both via a single `isinstance` check.

**`SingleCausalForecaster` additions:**

- `freeze_structural_params` / `freeze_reconstruction_params` — read from config in `__init__`, applied in `on_fit_start` via `requires_grad_(False)` on the appropriate param group.
- Gate bias annealing (block 5 in `on_train_epoch_start`) — linearly drifts `gate_bias` on all decoder layers. For `ToeplitzAttention` (Parameter) uses `.fill_()`; for `SigmoidCrossAttention` (float) uses direct assignment.

#### Example config snippet

```yaml
anm_training:
  starting_checkpoint: null
  stages:
    # ── H1: reconstruction warmup ─────────────────────────────────────────
    - name: recon_warmup
      max_epochs: 30
      lambda_hsic_cross: 0.0
      lambda_hsic_self: 0.0
      freeze_structural_params: true
      batch_key_dropout_p: 0.8
      eval_every_n_epochs: 5        # inline StageEvalCallback (cheap)
      eval_dag: true
      evaluation:                   # post-stage checkpoint-loading eval
        functions:
          - eval_anm_residual_hsic  # H1 diagnostic: per-edge HSIC heatmap

    # ── H2 / H3: structure phase at p=0.8 ────────────────────────────────
    - name: struct_phase_p08
      max_epochs: 20
      lambda_hsic_cross: 0.1
      lambda_recon: 0.0
      freeze_reconstruction_params: true
      batch_key_dropout_p: 0.8
      eval_every_n_epochs: 5
      eval_dag: true
      evaluation:
        functions:
          - eval_anm_residual_hsic
          - eval_train_metrics

    # ── H5: structure phase with gate-bias drift, p=0.4 ──────────────────
    - name: struct_bias_drift
      max_epochs: 30
      lambda_hsic_cross: 0.1
      freeze_reconstruction_params: true
      batch_key_dropout_p: 0.4
      use_gate_bias_annealing: true
      gate_bias_start: 0.0
      gate_bias_end: -20.0
      gate_bias_anneal_epochs: 30
      eval_dag: true
      evaluation:
        functions:
          - eval_anm_residual_hsic
          - eval_train_metrics

    # ── Final joint fine-tuning ───────────────────────────────────────────
    - name: joint_finetune
      max_epochs: 10
      batch_key_dropout_p: 0.0
      eval_dag: true
      evaluation:                   # classical post-training evals on final stage
        functions:
          - eval_train_metrics
          - eval_attention_scores   # DAG recovery metrics (soft Hamming, MEC)
          - eval_interventions      # ATE causal intervention tests
          - eval_anm_residual_hsic  # final residual-HSIC for comparison
```

Each stage writes its outputs to `<save_dir>/anm_stages/<idx>_<name>/`.  The `evaluation.functions` outputs land under `<stage_dir>/eval/<function_name>/` following the standard eval directory layout.
