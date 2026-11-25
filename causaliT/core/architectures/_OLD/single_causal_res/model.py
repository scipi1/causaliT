"""SingleCausalLayerRes: SVFA dual-residual variant of SingleCausalLayer.

Architecture summary
--------------------
Identical to ``SingleCausalLayer`` (single-decoder, S -> X, reversed
attention order) EXCEPT the decoder is the dual-residual variant from
``causaliT.core.architectures.single_causal_res.decoder`` and both the
cross- and self-attention layers are constructed with ``dual_value=True``.

This means each attention block updates BOTH streams of the SVFA pair
``(X_struct, X_val)`` via residual connections (see
``docs/SVFA_DUAL_RESIDUAL.md``), while keeping the value forecast head
reading from ``X_val`` only.

The class subclasses ``SingleCausalLayer`` to avoid duplicating embedding
and attention-construction logic; the only overrides are:

* ``factorization`` is forced to ``"svfa"`` (the dual-residual decoder is
  SVFA-only by design).
* ``_attn(...)`` injects ``dual_value=True``.
* The decoder stack is rebuilt using ``DualResidualDecoder`` /
  ``DualResidualDecoderLayer`` (reusing every parameter the parent
  __init__ already created).
* ``_forward_bypass`` is disabled (ANS bypass would need a structural
  bypass formulation; out of scope).
"""

import torch.nn as nn

from causaliT.core.architectures.single_causal.model import SingleCausalLayer
from causaliT.core.architectures.single_causal_res.decoder import (
    DualResidualDecoder,
    DualResidualDecoderLayer,
)


class SingleCausalLayerRes(SingleCausalLayer):
    """SingleCausalLayer + SVFA dual-residual decoder.

    All constructor kwargs are inherited from ``SingleCausalLayer``. The
    ``factorization`` kwarg is forced to ``"svfa"`` -- passing anything
    else raises ``ValueError``.
    """

    def __init__(self, *args, **kwargs):
        # Force SVFA: the dual-residual decoder requires the (X_struct, X_val)
        # tuple representation. Reject any other factorization explicitly so
        # mis-configurations surface early.
        factorization = kwargs.get("factorization", "svfa")
        if factorization != "svfa":
            raise ValueError(
                f"SingleCausalLayerRes requires factorization='svfa' "
                f"(got '{factorization}'). The dual-residual decoder is "
                f"SVFA-only by construction."
            )
        kwargs["factorization"] = "svfa"

        if kwargs.get("attention_bypass", False):
            raise NotImplementedError(
                "attention_bypass=True is not supported by "
                "SingleCausalLayerRes. Use SingleCausalLayer for ANS "
                "evaluation."
            )

        # Build the parent (which creates dual-value attentions thanks to
        # our overridden _attn below, and a ReversedDecoder we will swap).
        super().__init__(*args, **kwargs)

        # Swap the decoder for the dual-residual variant, REUSING every
        # parameter (attention modules, norms, FFN linears, dropouts) that
        # the parent already constructed so the parameter count and initial
        # state are unchanged relative to the parent build.
        old_decoder = self.decoder
        old_layers = list(old_decoder.layers)

        new_layers = []
        for old in old_layers:
            new = DualResidualDecoderLayer.__new__(DualResidualDecoderLayer)
            nn.Module.__init__(new)

            new.global_cross_attention = old.global_cross_attention
            new.global_self_attention = old.global_self_attention

            for label, att in (
                ("cross", new.global_cross_attention),
                ("self", new.global_self_attention),
            ):
                if not getattr(att, "dual_value", False):
                    raise RuntimeError(
                        f"Internal error: {label}-attention was built "
                        f"without dual_value=True in "
                        f"SingleCausalLayerRes._attn."
                    )

            # Reuse pre-norms and FFN.
            new.norm1 = old.norm1
            new.norm2 = old.norm2
            new.norm3 = old.norm3
            new.norm1_struct = old.norm1_struct
            new.norm2_struct = old.norm2_struct
            new.linear1 = old.linear1
            new.linear2 = old.linear2
            new.dropout_ff = old.dropout_ff
            new.dropout_attn_out = old.dropout_attn_out
            new.activation = old.activation

            new_layers.append(new)

        # Rebuild the decoder. We instantiate ``DualResidualDecoder`` with
        # ``emb_dropout=0.0`` and then directly attach the parent's
        # original ``emb_dropout`` module so the actual dropout rate is
        # preserved without having to read it back from ``nn.Dropout.p``.
        new_decoder = DualResidualDecoder(
            decoder_layers=new_layers,
            norm_layer=old_decoder.norm_layer,
            emb_dropout=0.0,
        )
        new_decoder.emb_dropout = old_decoder.emb_dropout
        self.decoder = new_decoder

    # ------------------------------------------------------------------
    # Inject dual_value=True into every AttentionLayer the parent builds.
    # ------------------------------------------------------------------
    def _attn(self, *args, **kwargs):
        kwargs["dual_value"] = True
        return super()._attn(*args, **kwargs)

    # ------------------------------------------------------------------
    # ANS bypass disabled (rejected in __init__ already).
    # ------------------------------------------------------------------
    def _forward_bypass(self, *args, **kwargs):
        raise NotImplementedError(
            "attention_bypass is not supported by SingleCausalLayerRes."
        )
