"""Edit script 3a: warmup phase — phase code, cfg parsing, bkd_managed."""
from pathlib import Path

AT = Path("causaliT/training/adaptive_trainer.py")


def replace_once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: anchor found {text.count(old)}x"
    return text.replace(old, new)


at = AT.read_text(encoding="utf-8")

at = replace_once(
    at,
    '_PHASE_CODE = {"reconstruct": 0, "structure": 1, "final_reconstruct": 2}',
    '_PHASE_CODE = {"reconstruct": 0, "structure": 1, "final_reconstruct": 2,\n'
    '               "warmup": 3}',
    "phase code",
)

at = replace_once(
    at,
    '        self.struct_cfg: Dict[str, Any] = _to_plain_container(ad.get("structure", {})) or {}',
    '        self.struct_cfg: Dict[str, Any] = _to_plain_container(ad.get("structure", {})) or {}\n'
    "\n"
    "        # Dedicated warmup phase (Arm 1): one big reconstruction-only phase\n"
    "        # under the periodic (cosine) BKD curriculum with BKD-coupled open\n"
    "        # gates, run ONCE before the alternating schedule.  Exits purely on\n"
    "        # its epoch budget (the schedule must complete).\n"
    '        self.warmup_cfg: Dict[str, Any] = _to_plain_container(ad.get("warmup", {})) or {}\n'
    '        self.warmup_enabled: bool = bool(self.warmup_cfg.get("enabled", False))\n'
    '        self.warmup_max_epochs: int = int(self.warmup_cfg.get("max_epochs", 100))\n'
    '        if self.start_phase == "warmup" and not self.warmup_enabled:\n'
    "            raise ValueError(\n"
    '                "start_phase=\'warmup\' requires "\n'
    '                "adaptive_training.warmup.enabled=true"\n'
    "            )",
    "warmup cfg parse",
)

at = replace_once(
    at,
    '        self._bkd_managed: bool = any(\n'
    '            "batch_key_dropout" in cfg\n'
    "            for cfg in (self.recon_cfg, self.struct_cfg, self.final_cfg)\n"
    "        )",
    '        self._bkd_managed: bool = any(\n'
    '            "batch_key_dropout" in cfg\n'
    "            for cfg in (self.recon_cfg, self.struct_cfg, self.final_cfg,\n"
    "                        self.warmup_cfg)\n"
    "        )",
    "bkd_managed",
)

AT.write_text(at, encoding="utf-8")
print("trainer edit 3a OK")
