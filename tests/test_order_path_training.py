"""Fast policy restoration and projection checks."""
from types import SimpleNamespace

import pytest
import torch

from causaliT.training.order_path_training import forward_policy, protect_weight


def test_projection():
    w = torch.tensor([1., 0.])
    r = torch.tensor([-2., 3.])
    out = protect_weight(w, r)
    torch.testing.assert_close(out, torch.tensor([0., 3.]))
    assert torch.dot(w + out, w) >= 0
    assert protect_weight(w, -r) is not None
    torch.testing.assert_close(protect_weight(w, -r), -r)
    torch.testing.assert_close(protect_weight(w * 0, r), r)


@pytest.mark.parametrize('sampling,training,expected', [
    ('soft', True, 'soft'), ('hard_st', True, 'hard_st'),
    ('hard_st', False, 'hard')])
def test_policy_restores_on_error(sampling, training, expected):
    from causaliT.core.modules.extra_layers import BKDMixin
    oa = SimpleNamespace(stage='structure', gate_mode='hard',
                         struct_gate_mode='hard', order_sampling='soft',
                         _bkd_phase_active=True, _bkd_step=torch.tensor(4))
    oa.set_stage_policy = lambda stage, **kw: (
        setattr(oa, 'stage', stage), setattr(oa, 'order_sampling', kw['order_sampling']))
    oa.set_bkd_phase_active = lambda active: BKDMixin.set_bkd_phase_active(oa, active)
    model = SimpleNamespace(order_attention=lambda: oa, modules=lambda: [oa], training=training)
    with forward_policy(model, structural=False):
        assert oa.stage == 'recon' and oa.order_sampling == 'none'
        assert oa._bkd_phase_active
        oa._bkd_step += 1
    assert oa._bkd_step == 5
    with pytest.raises(RuntimeError):
        with forward_policy(model, structural=True, order_sampling=sampling):
            assert oa.order_sampling == expected
            assert not oa._bkd_phase_active
            oa._bkd_step += 1
            raise RuntimeError('probe')
    assert oa._bkd_phase_active and oa._bkd_step == 5
    assert oa.stage == 'structure' and oa.order_sampling == 'soft'