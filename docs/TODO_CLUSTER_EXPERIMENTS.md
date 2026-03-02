# Cluster Experiments TODO

Experiments to run on the cluster from `experiments/single/scm6`.

## Status Key
- [ ] Not started
- [🔄] Running
- [x] Completed
- [❌] Failed

---

## Lie-based Self-Attention Experiments

### LieAttention × CausalCrossAttention

- [ ] `single_Lie_CC_scm6` - Baseline Lie×CC
- [ ] `single_Lie_CC_scm6_antisym` - Antisymmetric phi
- [ ] `single_Lie_CC_scm6_gated` - Gated phi
- [ ] `single_Lie_CC_scm6_indep` - Independent phi
- [ ] `single_Lie_CC_scm6_SVFA` - SVFA embeddings
- [ ] `single_Lie_CC_scm6_sweep_kl` - KL sweep

### LieAttention × Other Cross-Attention

- [ ] `single_Lie_PhiSM_scm6` - Lie self × PhiSoftMax cross
- [ ] `single_Lie_SM_scm6` - Lie self × ScaledDot cross

---

## PhiSoftMax-based Experiments

### PhiSoftMax × CausalCrossAttention

- [ ] `single_PhiSM_CC_scm6_antisym` - PhiSM self × CC cross (antisym)

### PhiSoftMax × PhiSoftMax

- [ ] `single_PhiSM_PhiSM_scm6` - Baseline PhiSM×PhiSM
- [ ] `single_PhiSM_PhiSM_scm6_antisym` - Antisymmetric phi
- [ ] `single_PhiSM_PhiSM_scm6_indep` - Independent phi
- [ ] `single_PhiSM_PhiSM_scm6_SVFA` - SVFA embeddings

---

## ScaledDotProduct-based Experiments

### ScaledDot × ScaledDot

- [ ] `single_SM_SM_scm6` - Baseline SM×SM
- [ ] `single_SM_SM_scm6_SVFA` - SVFA embeddings
- [ ] `single_SM_SM_scm6_hard` - Hard masks
- [ ] `single_SM_SM_scm6_sweep_kl` - KL sweep

---

## ToeplitzLie-based Experiments

### ToeplitzLieAttention × CausalCrossAttention

- [ ] `single_Toeplitz_CC_scm6` - Baseline Toeplitz×CC
- [ ] `single_Toeplitz_CC_scm6_SVFA` - SVFA embeddings
- [ ] `single_Toeplitz_CC_scm6_antisym` - Antisymmetric phi
- [ ] `single_Toeplitz_CC_scm6_gated` - Gated phi
- [ ] `single_Toeplitz_CC_scm6_indep` - Independent phi

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Lie-based | 8 | 0/8 |
| PhiSM-based | 5 | 0/5 |
| SM-based | 4 | 0/4 |
| Toeplitz-based | 4 | 0/4 |
| **Total** | **21** | **0/21** |

---

## Other Datasets (separate folder)

### experiments/single/scm7

- [ ] `single_Lie_CC_scm7` - Lie×CC on scm7 dataset

---

## Run Commands

```bash
# Run single experiment
python -m causaliT.cli train --exp_id experiments/single/scm6/single_Lie_CC_scm6 --cluster True

# Run with debug mode (for testing)
python -m causaliT.cli train --exp_id experiments/single/scm6/single_Lie_CC_scm6 --debug True
```

## Notes

- Naming consistency checked on: 2026-02-28
- All 21 experiments in scm6 are consistent ✓
