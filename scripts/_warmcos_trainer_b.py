"""Edit script 3b: warmup branch in _apply_bkd_cfg + schedule-shape kwargs."""
from pathlib import Path

AT = Path("causaliT/training/adaptive_trainer.py")


def replace_once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: anchor found {text.count(old)}x"
    return text.replace(old, new)


at = AT.read_text(encoding="utf-8")

at = replace_once(
    at,
    "        if not self._bkd_managed:\n"
    "            return\n"
    "\n"
    '        if phase == "reconstruct":\n'
    "            cfg = self.recon_cfg",
    "        if not self._bkd_managed:\n"
    "            return\n"
    "\n"
    '        if phase == "warmup":\n'
    "            cfg = self.warmup_cfg\n"
    '        elif phase == "reconstruct":\n'
    "            cfg = self.recon_cfg",
    "bkd warmup branch",
)

at = replace_once(
    at,
    "                if active:\n"
    "                    mod.set_bkd_schedule(\n"
    '                        p0=float(cfg["batch_key_dropout"]),\n'
    '                        p1=cfg.get("batch_key_dropout_final", None),\n'
    "                        annealing_batches=cfg.get(\n"
    '                            "batch_key_dropout_annealing_batches", None\n'
    "                        ),\n"
    "                    )",
    "                if active:\n"
    "                    shape_kw = {}\n"
    '                    if "bkd_schedule" in cfg:\n'
    '                        shape_kw["schedule"] = str(cfg["bkd_schedule"])\n'
    '                    if "bkd_p_base" in cfg:\n'
    '                        shape_kw["p_base"] = float(cfg["bkd_p_base"])\n'
    '                    if "bkd_amplitude" in cfg:\n'
    '                        shape_kw["amp"] = float(cfg["bkd_amplitude"])\n'
    '                    if "bkd_cycles" in cfg:\n'
    '                        shape_kw["cycles"] = float(cfg["bkd_cycles"])\n'
    "                    mod.set_bkd_schedule(\n"
    '                        p0=float(cfg["batch_key_dropout"]),\n'
    '                        p1=cfg.get("batch_key_dropout_final", None),\n'
    "                        annealing_batches=cfg.get(\n"
    '                            "batch_key_dropout_annealing_batches", None\n'
    "                        ),\n"
    "                        **shape_kw,\n"
    "                    )",
    "bkd schedule kwargs",
)

AT.write_text(at, encoding="utf-8")
print("trainer edit 3b OK")
