"""Smoke test: multi-layer FiLM head + eval-mode BKD."""
import torch

from causaliT.core.modules.mlp_head import PerNodeMLPHead
from causaliT.core.modules.gated_cross_attention import GatedCrossAttention

torch.manual_seed(0)
B, L, N, d_model, d_hidden, ctx = 4, 6, 6, 8, 16, 6

# --- 1) Legacy layout (n_layers=2, film) unchanged -----------------------
head2 = PerNodeMLPHead(d_model, 1, N, d_hidden=d_hidden,
                       film_context_dim=ctx, n_layers=2)
x = torch.randn(B, L, d_model)
var_ids = torch.randint(1, N + 1, (B, L))
context = torch.randn(B, L, ctx)
out2 = head2(x, var_ids, context=context)
assert out2.shape == (B, L, 1), out2.shape
# zero-init conditioner => identical to unconditioned decoder
head2b = PerNodeMLPHead(d_model, 1, N, d_hidden=d_hidden, film_context_dim=0)
head2b.load_state_dict(
    {k: v for k, v in head2.state_dict().items() if not k.startswith("film")},
    strict=False,
)
out2b = head2b(x, var_ids)
assert torch.allclose(out2, out2b, atol=1e-6), (out2 - out2b).abs().max()
print("n_layers=2 legacy-equivalence OK")

# --- 2) Deeper head with multi-layer FiLM --------------------------------
head4 = PerNodeMLPHead(d_model, 1, N, d_hidden=d_hidden,
                       film_context_dim=ctx, n_layers=4)
out4 = head4(x, var_ids, context=context)
assert out4.shape == (B, L, 1), out4.shape
n_blocks = len(head4.mlps[0])
assert n_blocks == 3 * (4 - 1) + 1, n_blocks  # 3 hidden blocks + output
# zero-init conditioner => any context gives identity modulation at init
out4_zeroctx = head4(x, var_ids, context=torch.zeros(B, L, ctx))
assert torch.allclose(out4, out4_zeroctx, atol=1e-6), \
    (out4 - out4_zeroctx).abs().max()
print("n_layers=4 multi-FiLM forward + zero-init identity OK")

# --- 3) Grad flows to conditioner ----------------------------------------
head4.zero_grad()
out4.sum().backward()
g = head4.film[-1].weight.grad
assert g is not None and g.abs().sum() > 0
print("conditioner gradient OK")

# --- 4) Eval-mode BKD on GatedCrossAttention ------------------------------
gca = GatedCrossAttention(
    attention_dropout=0.0, register_entropy=False, layer_name="t",
    init_tau=0.5, gamma=-1.1, zeta=1.1,
    batch_key_dropout=0.8, batch_key_dropout_min_keys=1,
    batch_key_dropout_deterministic=True,
    batch_key_dropout_eval=True, batch_key_dropout_eval_seed=7,
)
q = torch.randn(B, 5, d_model)
k = torch.randn(B, N, d_model)
v = torch.randn(B, N, d_model)

gca.eval()
out_e1, _, _ = gca(query=q, key=k, value=v)
keep1 = gca.last_bkd_keep.clone()
out_e2, _, _ = gca(query=q, key=k, value=v)
keep2 = gca.last_bkd_keep.clone()
assert keep1 is not None and keep2 is not None
assert int(keep1.sum()) == 1 and int(keep2.sum()) == 1  # min_keys=1, exact

# fresh module with same seed + reset counter reproduces the first mask
gca2 = GatedCrossAttention(
    attention_dropout=0.0, register_entropy=False, layer_name="t",
    init_tau=0.5, gamma=-1.1, zeta=1.1,
    batch_key_dropout=0.8, batch_key_dropout_min_keys=1,
    batch_key_dropout_deterministic=True,
    batch_key_dropout_eval=True, batch_key_dropout_eval_seed=7,
)
gca2.load_state_dict(gca.state_dict())
gca2.eval()
gca2._bkd_eval_step.zero_()
_, _, _ = gca2(query=q, key=k, value=v)
assert torch.equal(gca2.last_bkd_keep, keep1), "eval masks not reproducible"
print("eval-mode BKD OK (deterministic, exact count, reproducible)")

# eval disabled flag => no BKD at eval
gca.set_bkd_eval(False)
gca.eval()
gca(query=q, key=k, value=v)
assert gca.last_bkd_keep is None
print("set_bkd_eval(False) OK")

# training path still works and uses global RNG
gca.train()
gca(query=q, key=k, value=v)
assert gca.last_bkd_keep is not None
print("train-mode BKD OK")

print("ALL TESTS PASSED")
