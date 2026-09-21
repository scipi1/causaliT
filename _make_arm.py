from pathlib import Path

src = Path(
    r'c:\Users\ScipioneFrancesco\Documents\Projects\causaliT\experiments\6_INVESTIGATIONS\HSIC_CONSTRAINT\adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film\config.yaml'
).read_text(encoding='utf-8')

new_dir = Path(
    r'c:\Users\ScipioneFrancesco\Documents\Projects\causaliT\experiments\6_INVESTIGATIONS\HSIC_CONSTRAINT\adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film2_evalbkd'
)
new_dir.mkdir(parents=True, exist_ok=True)

# ---- header block --------------------------------------------------------
head_start = src.index('# ===========================================================================')
head_end = src.index('experiment:')
new_head = '''# ===========================================================================
# HSIC_CONSTRAINT / adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film2_evalbkd
# ===========================================================================
# VARIANT of adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film with TWO
# changes addressing the min_keys=1 failure mode (val R2 NEGATIVE at
# epoch 100 in the parent arm):
#
# 1) RUNG-AWARE VALIDATION (batch_key_dropout_eval: true): BKD is
#    training-only in the parent arm, so at rung 0 (min_keys=1) the decoder
#    and the FiLM conditioner are trained exclusively on single-key inputs
#    while validation runs with the FULL dense gate row (~20 keys) - both
#    the value stream (sum of ~20 gated values vs 1) and the FiLM context
#    (dense vs one-hot) are far out of the training distribution, and the
#    FiLM gamma multiplies hidden activations, so OOD extrapolation can
#    produce worse-than-mean predictions (negative R2) even when the
#    rung-0 fit itself is fine.  Eval-mode BKD applies the CURRENT rung's
#    exact key budget in eval with a dedicated seeded generator
#    (batch_key_dropout_eval_seed: 12345, mask index = eval-forward
#    counter): deterministic across epochs, no training-RNG consumption.
#    val_x_mae / val_hsic (and the phase-transition monitor) now measure
#    the regime actually being trained at each rung.
#
# 2) MULTI-LAYER FiLM (per_node_output_layers: 4): the parent arm applies
#    FiLM at a SINGLE hidden activation of a 2-layer per-node head.  Per
#    Perez et al. 2018, feature-wise modulation is applied at EVERY layer;
#    here each per-node decoder is deepened to 4 linear layers (hidden 64)
#    and the shared conditioner outputs a (gamma_k, beta_k) pair per hidden
#    activation (packed, zero-init last layer => identity at step 0).
#    n_layers=2 remains bit-compatible with the parent arm's checkpoints.
#
# FALSIFIABLE PREDICTION vs _bigmlp_film: val_x_mae / val R2 at rung 0
# becomes non-negative and tracks train fit (eval-regime match), and the
# deeper FiLM switch lowers the warmup residual floor at rungs 1-3.  If
# train-side R2 at rung 0 is ALSO ~0 (check metrics.csv), the remaining
# suspect is whole-batch single-key interference (one key identity per
# optimizer step shared across all 1024 samples) -> consider per-sample
# key sampling next.
#
#   python -m causaliT.cli train --exp_id 6_INVESTIGATIONS/HSIC_CONSTRAINT/adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film2_evalbkd
# ===========================================================================
'''
src = new_head + src[head_end:]

# ---- experiment: per_node_output_layers ----------------------------------
old = '''  per_node_output: true
  per_node_output_hidden: 64'''
new = '''  per_node_output: true
  per_node_output_hidden: 64
  # THE VARIANT (2): deeper per-node decoder with multi-layer FiLM.
  per_node_output_layers: 4'''
assert src.count(old) == 1
src = src.replace(old, new)

# ---- model.kwargs: bkd eval + pass-through --------------------------------
old = '''    batch_key_dropout_min_keys: 0
    batch_key_dropout_deterministic: true'''
new = '''    batch_key_dropout_min_keys: 0
    batch_key_dropout_deterministic: true
    # THE VARIANT (1): rung-aware validation - apply the current rung's key
    # budget in eval mode with a seeded generator (see header).
    batch_key_dropout_eval: true
    batch_key_dropout_eval_seed: 12345'''
assert src.count(old) == 1
src = src.replace(old, new)

old = '''    per_node_output: ${experiment.per_node_output}
    per_node_output_hidden: ${experiment.per_node_output_hidden}'''
new = '''    per_node_output: ${experiment.per_node_output}
    per_node_output_hidden: ${experiment.per_node_output_hidden}
    per_node_output_layers: ${experiment.per_node_output_layers}'''
assert src.count(old) == 1
src = src.replace(old, new)

(new_dir / 'config.yaml').write_text(src, encoding='utf-8')
print('wrote', new_dir / 'config.yaml')
