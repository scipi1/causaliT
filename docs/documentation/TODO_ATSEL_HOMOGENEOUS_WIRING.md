# Wiring `homogeneous_nodes` end-to-end (AttentionSelector) — **COMPLETE**

`AttentionSelectorLayer` (`causaliT/core/architectures/attention_selector/model.py`) was
harmonized with `SelfSelectorLayer`:

* new kwarg **`homogeneous_nodes: bool = False`**;
* `True` → the S/X prior is dropped. `[S ; X]` is ONE set of `N = L_S + L_X` nodes, each a
  value-blanked **query** and an actual-value **key/value**. ONE square block built from
  `self_attention_type` but stored as `self.attention` (canonical attribute → score sparsity /
  gradient routing / freezing / centroid init keep working); `self.self_attention is None`,
  `split_xx is False`; posterior `(B, N, N)`; head predicts `(B, N, out_dim)`; the cross
  `attention_type` is IGNORED;
* `False` → unchanged split mode (`attention` = S→X cross, `self_attention` = X→X);
* **`self_attention_type` is MANDATORY in both modes** (legacy cross-only variant,
  `combined_mask` and `_build_combined_mask()` were removed → `None` raises `ValueError`);
* `forward_with_actual(..., s_blanked=None)` — required iff `homogeneous_nodes=True`;
* masks: homogeneous → `homogeneous_mask = 1 - eye(N)`; split → `cross_mask` + `self_mask`;
* oracle works in BOTH modes (split slices the `(L_X, L_S+L_X)` GT; homogeneous needs the
  square `(N, N)` GT);
* S-side tables `query_embed_S`, `gain_q_embed_S`, `val_q_id_embed_S`; `is_gated` forced `True`;
* `shared_query` / `shared_key` raise `ValueError` with `homogeneous_nodes=True`;
* helpers: shape-aware `split_attention()`, new `split_attention_blocks()` and `source_scores()`;
* public attrs `self.N`, `self.homogeneous_nodes`.

---

## Wiring (all steps DONE)

* **Step 7.1 — test suite unblocked.** Every fixture/factory that builds an
  `AttentionSelectorLayer` or an AttentionSelector config now passes
  `self_attention_type` (mandatory), and the assertions that assumed the removed single
  combined block were re-pointed (`combined_mask` → `cross_mask`/`self_mask`; X→X gate →
  `model.self_attention.inner_attention`; split-mode oracle no longer raises
  `NotImplementedError`).
* **Step 1 — `causaliT/training/forecasters/attention_selector_forecaster.py`**
  * `self.homogeneous_nodes` + `self.N` stored next to `S_seq_len`/`X_seq_len`;
  * `forward` builds `s_blanked` (value column zeroed) and passes it to `forward_with_actual`;
  * `_load_combined_oracle_mask` assembles the **square `(N, N)`** GT adjacency in
    homogeneous mode (S rows all-zero, X rows `[dec_cross | dec_self]`);
  * `_step` target becomes `cat([S_values, X_values], dim=1)` → `(B, N)`, so MSE, the
    torchmetrics, residuals and the ANM diagnostics all follow the N-row layout;
  * HSIC candidate-parent set is `x_target` itself in homogeneous mode (already all N nodes);
  * NOTEARS uses the **full** `(N, N)` score tensor in homogeneous mode
    (`score_tensor[:, S_seq_len:]` only in split mode);
  * the interference gate reads `self_attention_type` in homogeneous mode.
* **Step 2 — `causaliT/training/gradient_routing.py`**: `"query_embed_X"` → prefix
  `"query_embed"`, so `query_embed_S` is routed STRUCTURAL too (verified: no reconstruction
  parameter name contains `query_embed`).
* **Step 3 — `causaliT/evaluation/predictors/attention_selector_predictor.py`**:
  `_process_forward_output` slices the X rows (`pred[:, L_S:, ...]`) in homogeneous mode so
  `pred_x` still aligns with the X ground truth, and exposes the S rows as `pred_s`.  The
  attention dict is forwarded unchanged (the square shape is handled in `eval_dag_query`);
  `_forward` goes through `AttentionSelectorForecaster.forward`, which builds `s_blanked`
  itself, so no extra plumbing was needed.
* **Step 4 — `causaliT/evaluation/eval_funs/helpers/eval_dag_query.py`**:
  `split_combined_attention` handles the square `(N, N)` case (select child rows `[L_S:, :]`,
  then split the columns). Resolution order documented; `L_S > 0` guards the `N == L_X`
  ambiguity.
* **Step 5 — `causaliT/training/anm_staged_trainer.py`**: `_compute_score_margin` is
  shape-aware — square `(N, N)` score tensors are row-sliced to the X children first.
* **Step 6 — `causaliT/config/templates/config_attention_selector.yaml`**: new
  `experiment.homogeneous_nodes: false` (documented: `attention_type` ignored,
  `self_attention_type` required, `shared_query`/`shared_key` must be false) wired to
  `model.kwargs.homogeneous_nodes`.  `experiment.self_attention_type` default changed
  `null` → `"GatedSelfAttention"` because the legacy cross-only variant was removed and
  `null` now raises `ValueError`.
* **Step 7.2 — `tests/test_atsel_homogeneous.py` (new, 51 tests)** covering: construction
  (`self_attention is None`, `split_xx is False`, `attention.inner_attention` is the
  self-attention class, `homogeneous_mask` shape `(N, N)` with zero diagonal); forward shapes
  `pred (B,N,1)` / `attn (B,N,N)` with zero diagonal; the three `ValueError`s (missing
  `s_blanked`, `shared_query`/`shared_key`, `self_attention_type=None`);
  `split_attention` / `split_attention_blocks` / `source_scores` consistency with
  `attn[:, L_S:, :]`; structural-key orthogonality for `orthogonal_fixed` /
  `orthogonal_learnable` (also under the shared `W_K` with
  `key_projection_type="orthogonal"`); feature smoke tests (SVFA, free query +
  `query_centroid_init` writing BOTH `query_embed_X`/`_S`, value-structure (query) injection,
  BKD, learnable query norm); gradient flow into the S-side tables; the square `(N, N)`
  oracle; and a 2-step end-to-end run through `AttentionSelectorForecaster`.
  Plus the two asserted-in-plan extras: `query_embed_S` lands in the structural group
  (`tests/test_gradient_routing_orthogonal.py`) and a square-input case in
  `tests/test_dag_query.py`.
* **Item 2 — live experiment configs.** No `config_atsel.yaml` under `experiments/**` set
  `self_attention_type: null`, so nothing was broken by the new mandatory kwarg; the one live
  arm (`experiments/6_INVESTIGATIONS/Q_NORM/baseline_learn_centroid_init_1/config_atsel.yaml`)
  now carries an explicit, documented `experiment.homogeneous_nodes: false` wired to
  `model.kwargs.homogeneous_nodes` (it does not inherit the template).  Its
  `shared_query`/`shared_key: true` are noted as incompatible with homogeneous mode.
* **Item 3 — forecaster docs + diagnostics.** The
  `attention_selector_forecaster.py` module docstring now documents BOTH topologies (posterior
  / target shapes, `s_blanked`, the square oracle, full-matrix NOTEARS, the HSIC source and the
  `query_embed` routing prefix).  `split_attention_blocks()`, `source_scores()` and a
  one-forward `get_diagnostic_blocks()` are exposed on the forecaster as thin pass-throughs, and
  `get_split_attention()` documents that it works in both modes.
* **Step 8 — `self_selector` DEPRECATED (kept functional).** `SelfSelectorLayer` emits a
  `DeprecationWarning` on construction and its module/package/forecaster docstrings point at
  `AttentionSelectorLayer(homogeneous_nodes=True)` / `AttentionSelectorForecaster`, with a
  migration recipe (`model_object`, `homogeneous_nodes: true`,
  `self_attention_type: GatedSelfAttention`, split `ds_embed` → `ds_embed_S`/`ds_embed_X` — no
  var-id re-indexing needed — and leave `shared_query`/`shared_key` false).  Rationale: the
  only remaining difference is one shared embedding table over a unified var-id namespace vs.
  two `embedding_S`/`embedding_X` instances, mathematically equivalent for the orthogonal
  schemes.  Nothing was deleted, so existing checkpoints/configs keep loading.

## Regression status

`python -m pytest tests -q` → **524 passed, 3 skipped**.  The 5 collection errors are the
environment-only `PermissionError` on pytest's shared temp dir
(`AppData\Local\Temp\pytest-of-…`); re-running those same files with
`--basetemp=.pytest_tmp` gives **5 passed, 5 skipped**, so config-consistency and the
all-models training smoke test are green too.

## Possible follow-ups (not required)

* Actually train a homogeneous arm end-to-end and check whether `source_scores()` recovers
  the S/X partition the model was not given (that is the scientific payoff of the mode).
* Once such a run is validated, decide whether to physically delete
  `causaliT/core/architectures/self_selector` + `self_selector_forecaster.py` (currently only
  deprecated) and migrate `tests/test_self_selector.py` /
  `tests/test_value_structure_injection.py` onto the homogeneous AttentionSelector.
