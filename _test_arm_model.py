"""Instantiate AttentionSelectorLayer from the new arm config and run
train/eval forwards with synthetic tensors (CPU), checking the eval-BKD
context path end to end."""
import torch
from omegaconf import OmegaConf

from causaliT.core.architectures.attention_selector.model import (
    AttentionSelectorLayer,
)

cfg = OmegaConf.load(
    r'experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/'
    r'adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film2_evalbkd/config.yaml'
)
kw = OmegaConf.to_container(cfg.model.kwargs, resolve=True)

n_source, n_input = 2, 18
N = n_source + n_input
S_seq_len, X_seq_len = n_source, n_input
num_emb_S, num_emb_X = n_source + 1, n_input + 1  # 1-indexed IDs, 0=pad

kw['device'] = 'cpu'
if kw.get('d_ff') is None:
    kw['d_ff'] = int(kw['d_model'] * 4.0)
if kw.get('d_qk') is None:
    kw['d_qk'] = int(kw['d_model'] * 1.0)
if kw.get('query_fanin_scale') == 'auto':
    kw['query_fanin_scale'] = 4.0
if kw.get('init_edge_offset') == 'auto':
    kw['init_edge_offset'] = 0.0
kw['S_seq_len'] = S_seq_len
kw['X_seq_len'] = X_seq_len
# Resolve the embedding specs that reference data.* interpolations.
for spec in (kw['ds_embed_S'], kw['ds_embed_X']):
    spec['setting']['sparse_grad'] = False
for mod in kw['ds_embed_S']['modules']:
    if mod.get('idx') is None:
        mod['idx'] = 0 if mod['label'] == 'value' else 1
    k = mod.get('kwargs', {})
    if 'num_variables' in k and k['num_variables'] is None:
        k['num_variables'] = S_seq_len
    if 'num_embeddings' in k and k['num_embeddings'] is None:
        k['num_embeddings'] = num_emb_S
for mod in kw['ds_embed_X']['modules']:
    if mod.get('idx') is None:
        mod['idx'] = 0 if mod['label'] == 'value' else 1
    k = mod.get('kwargs', {})
    if 'num_variables' in k and k['num_variables'] is None:
        k['num_variables'] = X_seq_len
    if 'num_embeddings' in k and k['num_embeddings'] is None:
        k['num_embeddings'] = num_emb_X

model = AttentionSelectorLayer(**kw)
print('model built; forecaster:', type(model.forecaster).__name__)
print('per-node head n_layers:', model.forecaster.n_layers)
assert model.forecaster.n_layers == 4
assert model.forecaster.film is not None
inner = model.attention.inner_attention
print('eval-BKD flag:', inner._bkd_eval)
assert inner._bkd_eval is True

# Simulate the ladder: rung 0 = min_keys 1, p = 1.
for layer in (model.attention, getattr(model, 'self_attention', None)):
    if layer is None:
        continue
    ia = layer.inner_attention
    if hasattr(ia, 'set_bkd_sampling'):
        ia.set_bkd_sampling(min_keys=1, deterministic=True)
    if hasattr(ia, 'set_bkd_schedule'):
        ia.set_bkd_schedule(1.0, 1.0, None)

B = 8
# feature layout: column 0 = value, column 1 = variable id (1-indexed)
s = torch.zeros(B, S_seq_len, 2)
s[:, :, 0] = torch.randn(B, S_seq_len)
s[:, :, 1] = torch.arange(1, S_seq_len + 1).float()
x = torch.zeros(B, X_seq_len, 2)
x[:, :, 0] = torch.randn(B, X_seq_len)
x[:, :, 1] = torch.arange(1, X_seq_len + 1).float()
x_b = x.clone()
x_b[:, :, 0] = 0.0  # blanked query values
s_b = s.clone()
s_b[:, :, 0] = 0.0  # homogeneous mode: S is also a query

model.train()
pred, att, aux = model.forward_with_actual(s, x_b, x, s_blanked=s_b)
print('train pred:', tuple(pred.shape), 'bkd keep sum:',
      int(inner.last_bkd_keep.sum()))
assert int(inner.last_bkd_keep.sum()) == 1  # rung 0: exactly one key

model.eval()
with torch.no_grad():
    pred_e, _, _ = model.forward_with_actual(s, x_b, x, s_blanked=s_b)
keep_e = inner.last_bkd_keep
print('eval pred:', tuple(pred_e.shape), 'eval bkd keep sum:',
      None if keep_e is None else int(keep_e.sum()))
assert keep_e is not None and int(keep_e.sum()) == 1, \
    'eval-BKD did not apply the rung budget'
# FiLM context reflects the eval mask: exactly one nonzero entry per row
ctx_used = model.attention.inner_attention.last_applied_A
# at most one nonzero (the kept key's gate value can itself clamp to 0)
assert (ctx_used[0, 0] != 0).sum().item() <= 1
assert torch.isfinite(pred_e).all()

model.train()
pred, _, _ = model.forward_with_actual(s, x_b, x, s_blanked=s_b)
pred.float().pow(2).mean().backward()
print('backward OK')
print('ALL ARM TESTS PASSED')
