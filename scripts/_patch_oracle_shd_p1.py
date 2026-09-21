"""Patch: hsic_constraint.source = hsic | oracle_shd in AttentionSelectorForecaster.

oracle_shd monitors the EXPECTED SHD to the ground-truth DAG (mean per-pair
misclassification of the directed gate posterior) instead of the HSIC -- a
validation test simulating a perfect causal estimator on the same dual-ascent
machinery.  The GT is loaded into a dedicated buffer that never reaches
forward; use_hard_masks / use_oracle_attention are hard-error rejected.
"""
from pathlib import Path

p = Path("causaliT/training/forecasters/attention_selector_forecaster.py")
src = p.read_text(encoding="utf-8")

# P1a: parse the source unconditionally (attribute must exist when disabled).
old = '        self.hsic_constraint_enabled = bool(hc_cfg.get("enabled", False))\n        if self.hsic_constraint_enabled:'
new = ('        self.hsic_constraint_enabled = bool(hc_cfg.get("enabled", False))\n'
       '        self.hsic_constraint_source = str(hc_cfg.get("source", "hsic"))\n'
       '        if self.hsic_constraint_enabled:')
assert src.count(old) == 1, "P1a anchor"
src = src.replace(old, new)

# P1b: source validation + leakage guards, before the tolerance parse.
old = '            self.hsic_tol = float(hc_cfg.get("tolerance", 0.0))'
new = '''            if self.hsic_constraint_source not in ("hsic", "oracle_shd"):
                raise ValueError(
                    "hsic_constraint.source must be 'hsic' or 'oracle_shd', "
                    f"got {self.hsic_constraint_source!r}"
                )
            if self.hsic_constraint_source == "oracle_shd":
                # Leakage guard: the GT must NEVER reach the forward pass.
                # ``oracle_combined_mask`` is intersected into the attention
                # hard mask whenever it is not None (regardless of the oracle
                # flag), so both GT-consuming modes must be off; the GT is
                # loaded into a dedicated ``oracle_shd_gt`` buffer.
                if config["training"].get("use_hard_masks", False):
                    raise ValueError(
                        "hsic_constraint.source='oracle_shd' requires "
                        "use_hard_masks=False (the loaded mask is intersected "
                        "into the attention hard mask even without oracle "
                        "mode, which would leak the GT into the forward pass)."
                    )
                if config["training"].get("use_oracle_attention", False):
                    raise ValueError(
                        "hsic_constraint.source='oracle_shd' requires "
                        "use_oracle_attention=False (GT leakage into the "
                        "forward pass)."
                    )
            self.hsic_tol = float(hc_cfg.get("tolerance", 0.0))'''
assert src.count(old) == 1, "P1b anchor"
src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("P1 ok")
