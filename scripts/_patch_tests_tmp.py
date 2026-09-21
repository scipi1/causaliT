from pathlib import Path
p = Path("tests/test_hsic_constraint.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    "import numpy as np\nimport pandas as pd",
    "import tempfile\n\nimport numpy as np\nimport pandas as pd",
)
s = s.replace(
    "def _oracle_cfg(tmp_path, **hc_overrides):",
    "def _oracle_cfg(tmp_root, **hc_overrides):",
)
s = s.replace("    ds = tmp_path / \"dummy\"", "    ds = Path(tmp_root) / \"dummy\"")
s = s.replace(
    "    return cfg, str(tmp_path), np.concatenate([cross, selfm], axis=1)",
    "    return cfg, tmp_root, np.concatenate([cross, selfm], axis=1)",
)
for name, args in [
    ("test_hard_masks_leakage_guard", ""),
    ("test_oracle_attention_leakage_guard", ""),
    ("test_gt_loaded_and_never_in_forward_path", ""),
    ("test_ema_tracks_expected_shd_not_hsic", ""),
    ("test_gradients_flow_through_oracle_shd", ", dual_init=1.0"),
]:
    s = s.replace(
        f"def {name}(self, tmp_path):",
        f"def {name}(self):",
    )
    s = s.replace(
        f"_oracle_cfg(tmp_path{args})",
        f"_oracle_cfg(tempfile.mkdtemp(){args})",
    )
p.write_text(s, encoding="utf-8")
print("patched")
