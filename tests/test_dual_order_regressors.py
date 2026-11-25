"""Independent head routing and reversed guided-BKD checks."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from causaliT.core.architectures.attention_selector.model import AttentionSelectorLayer
from causaliT.core.modules.mlp_head import PerNodeMLPHead
from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
from causaliT.training.core.gradient_routing import _is_structural_param


@pytest.mark.parametrize('orientation', ['causal', 'anticausal'])
def test_complete_bank_routing_and_gradient_isolation(orientation):
    causal = PerNodeMLPHead(d_model=4, out_dim=1, num_variables=3,
                           d_hidden=8, film_context_dim=3, n_layers=2)
    anti = deepcopy(causal)
    oa = SimpleNamespace(orientation=orientation)
    model = SimpleNamespace(dual_order_regressors=True, forecaster=causal,
                            anticausal_forecaster=anti, order_attention=lambda: oa)
    selected = AttentionSelectorLayer._order_prediction_head(model)
    x = torch.randn(2, 3, 4)
    ids = torch.arange(1, 4).expand(2, -1)
    selected(x, ids, context=torch.randn(2, 3, 3)).square().mean().backward()
    inactive = anti if orientation == 'causal' else causal
    assert all(p.grad is None for p in inactive.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in selected.parameters())
    assert not ({p.data_ptr() for p in causal.parameters()} & {p.data_ptr() for p in anti.parameters()})
    assert all(not _is_structural_param('anticausal_forecaster.' + n) for n, _ in anti.named_parameters())


@pytest.mark.parametrize('selector', [False, True])
def test_guidance_reverses_selected_edges(selector):
    oa = OrderFactorizedSelfAttention(num_nodes=4, batch_key_dropout=0.5,
                                     bkd_order_guided={'enabled': True, 'start_epoch': 0,
                                                      'anneal_epochs': 0})
    oa.set_bkd_guide_epoch(1)
    mask = torch.triu(torch.ones(4, 4), diagonal=1)
    if selector:
        existence = torch.ones(4, 4)
        existence[0, 2] = existence[2, 0] = 0
        mask *= existence
    causal = oa.bkd_guided_drop(0.5, mask)
    reverse = oa.bkd_guided_drop(0.5, mask.T)
    torch.testing.assert_close(reverse, causal.T)
    assert reverse[0, 1] < reverse[1, 0]
    if selector:
        assert reverse[0, 2] == reverse[2, 0]