"""NoiseAwareSingleCausalLayerRes: noise-aware SVFA dual-residual variant.

Architecture summary
--------------------
Identical to ``NoiseAwareSingleCausalLayer`` (single-decoder, S → X,
reversed attention order, probabilistic output with ambient + reading noise)
EXCEPT the decoder is the dual-residual variant from
``causaliT.core.architectures.noise_aware_res.decoder`` and both the
cross- and self-attention layers are constructed with ``dual_value=True``.

Dual-residual semantics (per layer)
------------------------------------
    Cross-attention:
        out_val, out_struct = CrossAtt(...)        # dual_value=True
        H_det    = X_val    + dropout(out_val)
        X_struct = X_struct + dropout(out_struct)  ← NEW vs noise_aware

    Noise injection:
        H = AmbientNoise(H_det)                    ← unchanged

    Self-attention:
        out_val, out_struct = SelfAtt(...)         # dual_value=True; V_val = H
        U        = H       + dropout(out_val)
        X_struct = X_struct + dropout(out_struct)  ← NEW vs noise_aware

    Feedforward (value stream only):
        X_val_out = U + FF(LN(U))

Output
------
The probabilistic output head is UNCHANGED: ``(μ, log_var) = head(X_val)``
where X_val comes from the value stream only.

Implementation
--------------
This class subclasses ``NoiseAwareSingleCausalLayer`` and:

1. Forces ``factorization="svfa"`` (dual-residual is SVFA-only).
2. Overrides ``_attn`` to inject ``dual_value=True`` into every
   ``AttentionLayer`` built by ``super().__init__``.
3. After ``super().__init__()`` swaps each ``NoiseAwareReversedDecoderLayer``
   for a ``NoiseAwareDualResDecoderLayer`` by re-using ALL existing parameters
   (attention modules, norms, FFN linears, ambient noise layer, dropouts),
   so the parameter count and initial state are unchanged.
4. Wraps the layer stack in ``NoiseAwareDualResDecoder`` which preserves the
   ``inject_noise`` forwarding of the original decoder.
"""

import torch.nn as nn

from causaliT.core.architectures.noise_aware.model import NoiseAwareSingleCausalLayer
from causaliT.core.architectures.noise_aware_res.decoder import (
    NoiseAwareDualResDecoder,
    NoiseAwareDualResDecoderLayer,
)


class NoiseAwareSingleCausalLayerRes(NoiseAwareSingleCausalLayer):
    """NoiseAwareSingleCausalLayer + SVFA dual-residual decoder.

    All constructor kwargs are inherited from ``NoiseAwareSingleCausalLayer``.
    The ``factorization`` is always ``"svfa"`` — the dual-residual decoder
    requires the ``(X_struct, X_val)`` tuple representation.
    """

    def __init__(self, *args, **kwargs):
        # The dual-residual decoder is SVFA-only by construction; the parent
        # already hardcodes ``self.factorization = "svfa"`` so no extra check
        # is needed here — but we guard anyway for clarity.
        factorization = kwargs.get("factorization", "svfa")
        if factorization != "svfa":
            raise ValueError(
                f"NoiseAwareSingleCausalLayerRes requires factorization='svfa' "
                f"(got '{factorization}'). The dual-residual decoder is "
                f"SVFA-only by construction."
            )

        # Build the parent.  Our overridden _attn (below) is called during
        # super().__init__, so all attention layers are built with
        # dual_value=True from the start.
        super().__init__(*args, **kwargs)

        # ------------------------------------------------------------------
        # Swap every NoiseAwareReversedDecoderLayer → NoiseAwareDualResDecoderLayer
        # by reusing ALL existing sub-modules (no new parameters created).
        # ------------------------------------------------------------------
        old_decoder = self.decoder
        old_layers = list(old_decoder.layers)

        new_layers = []
        for old in old_layers:
            # Verify that dual_value was correctly injected.
            for label, att in (
                ("cross", old.global_cross_attention),
                ("self", old.global_self_attention),
            ):
                if not getattr(att, "dual_value", False):
                    raise RuntimeError(
                        f"Internal error: {label}-attention was built without "
                        f"dual_value=True in NoiseAwareSingleCausalLayerRes._attn."
                    )

            # Construct a new layer shell without calling __init__.
            new = NoiseAwareDualResDecoderLayer.__new__(NoiseAwareDualResDecoderLayer)
            nn.Module.__init__(new)

            # Re-attach all sub-modules from the old layer.
            new.global_cross_attention = old.global_cross_attention
            new.global_self_attention = old.global_self_attention
            new.ambient_noise = old.ambient_noise
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

        # Rebuild the decoder.  Use emb_dropout=0.0 then reattach the
        # original module to preserve the actual dropout rate.
        new_decoder = NoiseAwareDualResDecoder(
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
