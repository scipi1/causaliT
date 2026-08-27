"""Smoke test for the bkd_warmup_06_high_logit_init_nonorm arm.

Builds the forecaster from the arm's config (norm: none, use_final_norm: false,
score_at_init: null) and exercises:
  1. construction: no norm1/norm2 parameters in the state dict;
  2. the eval-mode forward + HSIC path (where the cluster job crashed);
  3. a train-mode forward + full backward (grads finite, reaching the queries).
"""
import numpy as np
import torch
from omegaconf import OmegaConf

from causaliT.training.config_utils import populate_seq_lengths_from_dataset
from causaliT.training.experiment_control import update_config
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_utils import hsic_cross_per_pair

import sys

CFG = sys.argv[1] if len(sys.argv) > 1 else (
    "experiments/6_INVESTIGATIONS/HSIC_OPT/bkd_warmup_06_high_logit_init_nonorm/config.yaml"
)
DATA = "data"

cfg = OmegaConf.load(CFG)
cfg = update_config(cfg)  # resolve d_ff / d_qk from the multipliers
cfg = populate_seq_lengths_from_dataset(cfg, DATA)
config = OmegaConf.to_container(cfg, resolve=True)

fm = AttentionSelectorForecaster(config, data_dir=DATA)
dev = "cuda" if torch.cuda.is_available() else "cpu"
fm = fm.to(dev)

# ---- 1. No shared norm parameters ------------------------------------------
norm_params = [n for n, _ in fm.named_parameters() if "norm1" in n or "norm2" in n]
print("norm1/norm2 parameters:", norm_params)
assert not norm_params, "shared norms must be gone"
m0 = fm.model.attention.inner_attention.query_norm_log_scale.exp()
print(f"M(0): mean {m0.mean().item():.4f} (expect 1.0 with score_at_init=null)")
assert torch.allclose(m0, torch.ones_like(m0)), "M(0) must be 1"

# ---- 2. Eval-mode forward + HSIC (the cluster crash path) -------------------
d = np.load(f"{DATA}/random_n20_k4_er_nonlinear_gaussian_s1/ds.npz")
S = torch.tensor(np.asarray(d["s"][:256]), dtype=torch.float32, device=dev)
X = torch.tensor(np.asarray(d["x"][:256]), dtype=torch.float32, device=dev)

fm.eval()
pred_x, attn, _ = fm.forward(data_source=S, data_intermediate=X)
assert torch.isfinite(pred_x).all(), "non-finite predictions in eval"
x_val = torch.cat([S[:, :, fm.val_idx], X[:, :, fm.val_idx]], dim=1)
x_target = torch.nan_to_num(x_val)
residuals = x_target.squeeze() - pred_x.squeeze()
hsic = hsic_cross_per_pair(
    x_target.squeeze(), residuals, sigma=fm.hsic_sigma,
    adaptive_bandwidth=fm.hsic_adaptive_bandwidth, mode=fm.hsic_mode,
    nhsic_epsilon=fm.nhsic_epsilon, source_kernel=fm.hsic_kernel_source,
    pair_mask=None,
)
print(f"eval HSIC: {hsic.item():.6f} (finite: {torch.isfinite(hsic).item()})")
assert torch.isfinite(hsic)

# ---- 3. Train-mode forward + backward ---------------------------------------
fm.train()
pred_x, attn, _ = fm.forward(data_source=S, data_intermediate=X)
residuals = x_target.squeeze() - pred_x.squeeze()
mse = torch.nn.functional.mse_loss(pred_x.squeeze(), x_target.squeeze())
hsic = hsic_cross_per_pair(
    x_target.squeeze(), residuals, sigma=fm.hsic_sigma,
    adaptive_bandwidth=fm.hsic_adaptive_bandwidth, mode=fm.hsic_mode,
    nhsic_epsilon=fm.nhsic_epsilon, source_kernel=fm.hsic_kernel_source,
    pair_mask=None,
)
loss = mse + config["training"]["lambda_hsic"] * hsic
loss.backward()
q = fm.model.query_embed_X.embedding.weight
print(f"train loss {loss.item():.4f} (mse {mse.item():.4f}, hsic {hsic.item():.6f})")
print(f"query grad: norm {q.grad.norm().item():.4g}, finite: {torch.isfinite(q.grad).all().item()}")
assert torch.isfinite(q.grad).all() and q.grad.norm() > 0

# ---- 4. Phase controller: BKD active with the same schedule in both phases --
from causaliT.training.adaptive_trainer import PhaseController
ctrl = PhaseController(config=config, data_dir=DATA, save_dir="_smoke_out",
                       cluster=True)
att = fm.model.attention.inner_attention
for phase in ("reconstruct", "structure", "final_reconstruct"):
    ctrl._apply_bkd_cfg(fm, phase)
    print(f"phase={phase:18s} bkd_active={att._bkd_phase_active} "
          f"p0={att._bkd_p0} p1={att._bkd_p1} anneal={att._bkd_anneal}")
    assert att._bkd_phase_active, f"BKD must be active in {phase}"
    assert att._bkd_p0 == 0.6 and att._bkd_p1 == 0.05 and att._bkd_anneal == 3000

print("SMOKE TEST PASSED")
