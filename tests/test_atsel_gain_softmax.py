"""Tests for the prior-softmax reconstruction gain (GainSoftmax).

Run with:  pytest tests/test_atsel_gain_softmax.py -v

Design under test
-----------------
The gated attentions apply the structure gate directly as the attention weight
(``A = z``).  The GainSoftmax stage (causaliT/core/modules/gain_softmax.py)
adds a reconstruction-driven gain that REDISTRIBUTES each row's gate mass
within the gate's own support::

    A = (1 - lambda) * z  +  lambda * n * z * exp(s) / D
    D = sum_k z_k * exp(s_k)         n = sum_k z_k (detached)

i.e. at ``lambda = 1`` the applied weight is ``n * softmax(log z + s)``: the
gate is the multiplicative softmax prior.  The invariants under test:

* ``lambda = 0``           -> bit-identical to the gate-only baseline;
* zero-init scores         -> identity at ANY lambda (smooth turn-on);
* ``z = 0``                -> ``A = 0`` EXACTLY (forbidden / gated-off edges);
* row mass preserved       -> ``sum_j A_ij == sum_j z_ij`` at every lambda;
* all-off row              -> zero row (no NaN), the gate-only behaviour;
* differentiable in ``z`` INCLUDING at ``z = 0`` (re-opening pressure);
* gain parameters are RECONSTRUCTION-routed (gradient-routing names);
* the trainer ramps ``lambda`` over the alternating schedule (global epoch,
  phase-agnostic - no separate final phase).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def tmp_path():
    """Workspace-local temp dir.

    The default pytest ``tmp_path`` fixture points at the system temp root,
    which is not readable in this environment (WinError 5).  Create the dir
    under the project instead and clean it up afterwards.
    """
    base = project_root / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

from causaliT.core.modules.gain_softmax import GainSoftmax
from causaliT.core.modules.gated_cross_attention import GatedCrossAttention
from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.training.gradient_routing import classify_parameters
from causaliT.training.adaptive_trainer import PhaseController


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1


# ===========================================================================
# Part 1 - GainSoftmax unit tests
# ===========================================================================


def _gate(B, L, S, seed=0):
    torch.manual_seed(seed)
    gate = torch.rand(B, L, S)
    gate[:, :, -1] = 0.0          # a forbidden column
    gate[0, 0, :] = 0.0           # an all-off row
    return gate


class TestGainSoftmaxUnit:
    def test_lambda_zero_is_identity(self):
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        gate = _gate(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        scores = torch.randn(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        out = g(gate, scores)
        assert torch.equal(out, gate)
        assert g.last_gain is None            # no diagnostics write when inert

    def test_zero_scores_identity_at_lambda_one(self):
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        g.set_gain_lambda(1.0)
        gate = _gate(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        out = g(gate, torch.zeros(BATCH, X_SEQ_LEN, S_SEQ_LEN))
        assert torch.allclose(out, gate, atol=1e-6)

    def test_exact_zeros_preserved(self):
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        g.set_gain_lambda(1.0)
        gate = _gate(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        scores = torch.randn(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        out = g(gate, scores)
        assert out[:, :, -1].abs().max() == 0.0     # forbidden column
        assert out[0, 0, :].abs().max() == 0.0      # all-off row
        assert torch.isfinite(out).all()

    def test_mass_preservation(self):
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        g.set_gain_lambda(1.0)
        gate = _gate(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        scores = torch.randn(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        out = g(gate, scores)
        assert torch.allclose(out.sum(-1), gate.sum(-1), atol=1e-5)

    def test_partial_lambda_interpolates(self):
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        gate = _gate(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        scores = torch.randn(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        g.set_gain_lambda(1.0)
        full = g(gate, scores)
        g.set_gain_lambda(0.5)
        half = g(gate, scores)
        assert torch.allclose(half, 0.5 * gate + 0.5 * full, atol=1e-6)

    def test_static_only_path(self):
        """gain_scores=None -> the zero-init static table alone (identity)."""
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        g.set_gain_lambda(1.0)
        gate = _gate(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        assert torch.allclose(g(gate, None), gate, atol=1e-6)

    def test_reopening_pressure_at_zero_gate(self):
        """A SELECTIVE loss rewarding a gated-off key gives its z=0 entry a
        finite, positive gradient through the prior (structure can re-open)."""
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        g.set_gain_lambda(1.0)
        gate = _gate(BATCH, X_SEQ_LEN, S_SEQ_LEN).requires_grad_(True)
        scores = torch.randn(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        w = torch.zeros(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        w[:, :, -1] = 1.0                       # reward the gated-off column
        (g(gate, scores) * w).sum().backward()
        assert gate.grad[0, 1, -1] > 0

    def test_static_logits_receive_gradient(self):
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        g.set_gain_lambda(1.0)
        gate = _gate(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        scores = torch.randn(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        w = torch.zeros(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        w[:, :, 0] = 1.0                        # reward an ON column
        (g(gate, scores) * w).sum().backward()
        assert g.gain_static_logits.grad is not None
        assert g.gain_static_logits.grad.abs().sum() > 0

    def test_set_gain_lambda_validates_range(self):
        g = GainSoftmax(X_SEQ_LEN, S_SEQ_LEN)
        with pytest.raises(ValueError):
            g.set_gain_lambda(1.5)


# ===========================================================================
# Part 2 - Inner-attention integration (GatedCrossAttention / GatedSelfAttention)
# ===========================================================================


def _proj(B, L, E):
    return torch.randn(B, L, E)


class TestGatedCrossGain:
    def test_gain_module_built(self):
        att = GatedCrossAttention(
            use_gain_softmax=True,
            gain_num_queries=X_SEQ_LEN,
            gain_num_keys=S_SEQ_LEN,
        )
        assert att.gain_softmax is not None
        assert att.gain_softmax.gain_static_logits.shape == (X_SEQ_LEN, S_SEQ_LEN)

    def test_requires_sizes(self):
        with pytest.raises(ValueError):
            GatedCrossAttention(use_gain_softmax=True)

    def test_lambda_zero_matches_no_gain(self):
        """At lambda=0 the gain path is bit-identical to the gate-only path."""
        att_g = GatedCrossAttention(
            use_gain_softmax=True, gain_num_queries=X_SEQ_LEN,
            gain_num_keys=S_SEQ_LEN,
        )
        att_g.eval()
        att_ref = GatedCrossAttention()
        att_ref.eval()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        scores = torch.randn(BATCH, X_SEQ_LEN, S_SEQ_LEN)
        out_g, _, _ = att_g(q, k, v, gain_scores=scores)
        out_ref, _, _ = att_ref(q, k, v)
        assert torch.allclose(out_g, out_ref, atol=1e-6)

    def test_lambda_one_at_init_is_identity(self):
        """Zero-init scores -> the gain is an exact identity even at lambda=1."""
        att = GatedCrossAttention(
            use_gain_softmax=True, gain_num_queries=X_SEQ_LEN,
            gain_num_keys=S_SEQ_LEN,
        )
        att.eval()
        att.gain_softmax.set_gain_lambda(1.0)
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        out_gain, _, _ = att(q, k, v, gain_scores=None)
        att.gain_softmax.set_gain_lambda(0.0)
        out_ref, _, _ = att(q, k, v)
        assert torch.allclose(out_gain, out_ref, atol=1e-6)

    def test_hard_mask_zeros_preserved_with_gain(self):
        att = GatedCrossAttention(
            use_gain_softmax=True, gain_num_queries=X_SEQ_LEN,
            gain_num_keys=X_SEQ_LEN,
        )
        att.eval()
        att.gain_softmax.set_gain_lambda(1.0)
        # Nonzero static logits + data scores: the mask must still win.
        with torch.no_grad():
            att.gain_softmax.gain_static_logits.uniform_(-1.0, 1.0)
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        mask = 1.0 - torch.eye(X_SEQ_LEN)       # no self-loops
        scores = torch.randn(BATCH, X_SEQ_LEN, X_SEQ_LEN)
        out, gate, _ = att(q, k, v, hard_mask=mask, gain_scores=scores)
        # The returned posterior keeps the masked zeros (DAG contract).
        diag = torch.diagonal(gate, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag))
        # And the applied weights are finite.
        assert torch.isfinite(out).all()


class TestGatedSelfGain:
    def test_lambda_one_at_init_is_identity(self):
        att = GatedSelfAttention(
            use_gain_softmax=True, gain_num_queries=X_SEQ_LEN,
            gain_num_keys=X_SEQ_LEN,
        )
        att.eval()
        att.gain_softmax.set_gain_lambda(1.0)
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        out_gain, _, _ = att(q, k, v, gain_scores=None)
        att.gain_softmax.set_gain_lambda(0.0)
        out_ref, _, _ = att(q, k, v)
        assert torch.allclose(out_gain, out_ref, atol=1e-6)

    def test_diagonal_stays_zero_with_gain(self):
        att = GatedSelfAttention(
            use_gain_softmax=True, gain_num_queries=X_SEQ_LEN,
            gain_num_keys=X_SEQ_LEN,
        )
        att.eval()
        att.gain_softmax.set_gain_lambda(1.0)
        with torch.no_grad():
            att.gain_softmax.gain_static_logits.uniform_(-1.0, 1.0)
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        scores = torch.randn(BATCH, X_SEQ_LEN, X_SEQ_LEN)
        out, p_directed, _ = att(q, k, v, gain_scores=scores)
        diag = torch.diagonal(p_directed, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag))
        assert torch.isfinite(out).all()


# ===========================================================================
# Part 3 - AttentionSelectorLayer wiring
# ===========================================================================


def _svfa_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _make_gain_model(use_gain_softmax=True, gain_data=True, **overrides):
    kwargs = dict(
        model="test_gain",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X),
        comps_embed_S="svfa",
        comps_embed_X="svfa",
        attention_type="GatedCrossAttention",
        self_attention_type="GatedSelfAttention",
        n_heads=1,
        dropout_emb=0.0,
        dropout_attn_out=0.0,
        dropout_ff=0.0,
        dropout_qkv=0.0,
        attention_dropout=0.0,
        activation="relu",
        norm="layer",
        use_final_norm=False,
        device="cpu",
        out_dim=1,
        d_ff=D_FF,
        d_model=D_MODEL,
        d_qk=D_QK,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        remove_query_projection=False,
        remove_key_projection=False,
        struct_embedding_type="standard_learnable",
        value_structure_injection="separate",
        value_structure_query_injection="separate",
        use_gain_softmax=use_gain_softmax,
        gain_data=gain_data,
    )
    kwargs.update(overrides)
    return AttentionSelectorLayer(**kwargs)


def _make_inputs():
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, VALUE_COL] = torch.randn(BATCH, S_SEQ_LEN)
    source[:, :, VAR_COL] = torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)

    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, VALUE_COL] = torch.randn(BATCH, X_SEQ_LEN)
    x_actual[:, :, VAR_COL] = torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)

    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked


class TestModelConstruction:
    def test_gain_modules_on_both_blocks(self):
        m = _make_gain_model()
        assert m.attention.inner_attention.gain_softmax is not None
        assert m.self_attention.inner_attention.gain_softmax is not None
        # Static tables sized (children, parents) per block.
        assert m.attention.inner_attention.gain_softmax.gain_static_logits.shape == (
            X_SEQ_LEN, S_SEQ_LEN)
        assert m.self_attention.inner_attention.gain_softmax.gain_static_logits.shape == (
            X_SEQ_LEN, X_SEQ_LEN)

    def test_gain_projections_zero_init_key(self):
        m = _make_gain_model()
        kproj = m.attention.gain_softmax_k_proj
        assert kproj is not None
        assert kproj.weight.abs().sum() == 0.0 and kproj.bias.abs().sum() == 0.0

    def test_no_gain_by_default(self):
        m = _make_gain_model(use_gain_softmax=False)
        assert m.attention.inner_attention.gain_softmax is None
        assert m.attention.gain_softmax_q_proj is None

    def test_static_only_arm_builds_no_projections(self):
        m = _make_gain_model(gain_data=False)
        assert m.attention.inner_attention.gain_softmax is not None
        assert m.attention.gain_softmax_q_proj is None
        assert m.attention.gain_softmax_k_proj is None

    def test_rejects_cross_only(self):
        with pytest.raises(ValueError, match="softmax prior"):
            _make_gain_model(self_attention_type=None)

    def test_rejects_commutator(self):
        with pytest.raises(ValueError, match="GatedSelfAttention"):
            _make_gain_model(self_attention_type="CommutatorSelfAttention")

    def test_rejects_missing_value_tables(self):
        with pytest.raises(ValueError, match="value_structure_injection"):
            _make_gain_model(value_structure_injection="none")
        with pytest.raises(ValueError, match="value_structure_query_injection"):
            _make_gain_model(value_structure_query_injection="none")


class TestModelForward:
    def test_lambda_zero_vs_lambda_one_identical_at_init(self):
        """Zero-init scores -> the gain is an exact identity at any lambda."""
        m = _make_gain_model()
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred0, attn0, _ = m.forward_with_actual(source, x_blanked, x_actual)
        n = m.set_gain_lambda(1.0)
        assert n == 2                                # both blocks updated
        pred1, attn1, _ = m.forward_with_actual(source, x_blanked, x_actual)
        assert torch.allclose(pred0, pred1, atol=1e-6)
        # The returned posterior is the GATE posterior in both cases (the DAG
        # contract is gain-independent).
        assert torch.allclose(attn0, attn1, atol=1e-6)

    def test_forward_with_active_gain_finite(self):
        m = _make_gain_model()
        m.eval()
        m.set_gain_lambda(1.0)
        # Make the data term nonzero: fill the key projection.
        with torch.no_grad():
            m.attention.gain_softmax_k_proj.weight.normal_(0.0, 0.1)
            m.self_attention.gain_softmax_k_proj.weight.normal_(0.0, 0.1)
        source, x_actual, x_blanked = _make_inputs()
        pred, attn, aux = m.forward_with_actual(source, x_blanked, x_actual)
        assert torch.isfinite(pred).all()
        # The X->X diagonal stays exactly zero (posterior contract).
        _, att_xx = m.split_attention(attn)
        diag = torch.diagonal(att_xx, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag))

    def test_gain_lambda_readback(self):
        m = _make_gain_model()
        assert m.gain_lambda() == 0.0
        m.set_gain_lambda(0.3)
        assert m.gain_lambda() == pytest.approx(0.3)


class TestGradientRouting:
    def test_gain_params_are_reconstruction(self):
        m = _make_gain_model()
        structural, reconstruction = classify_parameters(m)
        struct_ids = {id(p) for p in structural}
        recon_ids = {id(p) for p in reconstruction}

        gain_params = []
        for layer in (m.attention, m.self_attention):
            gain_params += list(layer.inner_attention.gain_softmax.parameters())
            if layer.gain_softmax_q_proj is not None:
                gain_params += list(layer.gain_softmax_q_proj.parameters())
                gain_params += list(layer.gain_softmax_k_proj.parameters())
        assert gain_params, "expected gain parameters to exist"
        for p in gain_params:
            assert id(p) in recon_ids, "gain params must be RECONSTRUCTION"
            assert id(p) not in struct_ids

    def test_recon_backward_reaches_gain_static_logits(self):
        m = _make_gain_model()
        m.train()
        m.set_gain_lambda(1.0)
        source, x_actual, x_blanked = _make_inputs()
        pred, _, _ = m.forward_with_actual(source, x_blanked, x_actual)
        targ = torch.zeros_like(pred)
        torch.nn.functional.mse_loss(pred, targ).backward()
        gs = m.attention.inner_attention.gain_softmax
        assert gs.gain_static_logits.grad is not None
        assert gs.gain_static_logits.grad.abs().sum() > 0


# ===========================================================================
# Part 4 - PhaseController: gain ramp + gain_train_structure
# ===========================================================================


class _FakeTrainer:
    def __init__(self, max_epochs=100):
        self.current_epoch = 0
        self.sanity_checking = False
        self.callback_metrics = {}
        self.optimizers = []
        self.should_stop = False
        self.max_epochs = max_epochs

    def save_checkpoint(self, path):
        pass


class _GainModule:
    """Fake forecaster exposing the gradient-routing groups + set_gain_lambda."""

    def __init__(self):
        self.training = True
        self._structural_params = [torch.nn.Parameter(torch.randn(3))]
        self._reconstruction_params = [torch.nn.Parameter(torch.randn(3))]
        self.lambda_history = []
        self.model = self          # the forecaster wraps the layer as .model

    def log(self, *args, **kwargs):
        pass

    def train(self):
        self.training = True

    def set_gain_lambda(self, value):
        self.lambda_history.append(float(value))
        return 2                   # two gain blocks updated


def _make_gain_controller(tmp_path, gain_cfg=None, total_budget=100):
    """PhaseController with the gain ramp keys at the adaptive_training level."""
    config = {
        "adaptive_training": {
            "monitor": "val_x_mae",
            "start_phase": "reconstruct",
            "max_cycles": 100,
            "eval_dag": False,
            "total_epoch_budget": total_budget,
            "reconstruct": {"max_epochs": 100, "min_epochs": 0,
                            "warmup_min_epochs": 0, "plateau_patience": 2,
                            "plateau_min_delta": 1e-4},
            "structure": {"max_epochs": 200, "drop_pct": 0.2, "drop_patience": 5},
            **(gain_cfg or {}),
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    return PhaseController(
        config=config, data_dir=str(tmp_path), save_dir=str(tmp_path), cluster=True,
    )


def _drive_epoch(controller, trainer, module, epoch, monitor=1.0):
    trainer.current_epoch = epoch
    trainer.callback_metrics = {"val_x_mae": monitor}
    controller.on_validation_epoch_end(trainer, module)


class TestGainPhaseController:
    def test_gain_keys_parsed(self, tmp_path):
        c = _make_gain_controller(tmp_path, {
            "gain_lambda_start": 40, "gain_lambda_ramp": 5,
            "gain_lambda_final": 0.7,
        })
        assert c.gain_lambda_start == 40
        assert c.gain_lambda_ramp == 5
        assert c.gain_lambda_final == pytest.approx(0.7)

    def test_gain_start_defaults_to_half_budget(self, tmp_path):
        c = _make_gain_controller(tmp_path, total_budget=200)
        assert c.gain_lambda_start == 100          # 0.5 * 200
        assert c.gain_lambda_ramp == 0             # jump
        assert c.gain_lambda_final == pytest.approx(1.0)

    def test_no_budget_no_start_is_noop(self, tmp_path):
        """Without total_epoch_budget AND without an explicit start, the ramp
        never fires (lambda stays wherever it was)."""
        config = {
            "adaptive_training": {
                "monitor": "val_x_mae", "start_phase": "reconstruct",
                "eval_dag": False,
                "reconstruct": {"max_epochs": 100},
                "structure": {"max_epochs": 200},
            },
            "model": {"model_object": "AttentionSelectorLayer"},
        }
        c = PhaseController(config=config, data_dir=str(tmp_path),
                            save_dir=str(tmp_path), cluster=True)
        assert c.gain_lambda_start is None
        trainer = _FakeTrainer(max_epochs=100)
        module = _GainModule()
        _drive_epoch(c, trainer, module, 10)
        assert module.lambda_history == []         # never touched

    def test_lambda_zero_before_start(self, tmp_path):
        c = _make_gain_controller(tmp_path, {
            "gain_lambda_start": 50, "gain_lambda_ramp": 10,
            "gain_lambda_final": 1.0,
        })
        trainer = _FakeTrainer(max_epochs=200)
        module = _GainModule()
        _drive_epoch(c, trainer, module, 10)       # before start
        assert module.lambda_history[-1] == 0.0

    def test_ramp_interpolates_and_saturates(self, tmp_path):
        c = _make_gain_controller(tmp_path, {
            "gain_lambda_start": 50, "gain_lambda_ramp": 10,
            "gain_lambda_final": 1.0,
        })
        trainer = _FakeTrainer(max_epochs=200)
        module = _GainModule()
        _drive_epoch(c, trainer, module, 50)       # start -> 0
        _drive_epoch(c, trainer, module, 55)       # mid -> 0.5
        _drive_epoch(c, trainer, module, 60)       # end -> 1.0
        _drive_epoch(c, trainer, module, 80)       # past -> stays 1.0
        assert module.lambda_history[-4] == 0.0
        assert module.lambda_history[-3] == pytest.approx(0.5)
        assert module.lambda_history[-2] == pytest.approx(1.0)
        assert module.lambda_history[-1] == pytest.approx(1.0)

    def test_ramp_is_phase_agnostic(self, tmp_path):
        """The ramp runs identically in reconstruct AND structure phases."""
        c = _make_gain_controller(tmp_path, {
            "gain_lambda_start": 50, "gain_lambda_ramp": 10,
            "gain_lambda_final": 1.0,
        })
        trainer = _FakeTrainer(max_epochs=200)
        module = _GainModule()
        c.current_phase = "reconstruct"
        _drive_epoch(c, trainer, module, 55)
        lam_recon = module.lambda_history[-1]
        c.current_phase = "structure"
        _drive_epoch(c, trainer, module, 55)       # same epoch, other phase
        lam_struct = module.lambda_history[-1]
        assert lam_recon == pytest.approx(0.5)
        assert lam_struct == pytest.approx(0.5)

    def test_zero_ramp_jumps_to_target(self, tmp_path):
        c = _make_gain_controller(tmp_path, {
            "gain_lambda_start": 50, "gain_lambda_ramp": 0,
            "gain_lambda_final": 0.8,
        })
        trainer = _FakeTrainer(max_epochs=200)
        module = _GainModule()
        _drive_epoch(c, trainer, module, 49)       # before start -> 0
        _drive_epoch(c, trainer, module, 50)       # at start -> target
        assert module.lambda_history[-2] == 0.0
        assert module.lambda_history[-1] == pytest.approx(0.8)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
