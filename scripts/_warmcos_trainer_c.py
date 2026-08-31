"""Edit script 3c: _apply_phase warmup branch + open-gate call + the
_apply_open_gate_cfg method."""
from pathlib import Path

AT = Path("causaliT/training/adaptive_trainer.py")


def replace_once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: anchor found {text.count(old)}x"
    return text.replace(old, new)


at = AT.read_text(encoding="utf-8")

at = replace_once(
    at,
    '        if phase in ("reconstruct", "final_reconstruct"):',
    '        if phase in ("reconstruct", "final_reconstruct", "warmup"):',
    "apply_phase freeze branch",
)
at = replace_once(
    at,
    "            mask_cfg = (\n"
    '                self.recon_cfg if phase == "reconstruct"\n'
    "                else {**self.recon_cfg, **self.final_cfg}\n"
    "            )",
    "            mask_cfg = (\n"
    '                self.recon_cfg if phase in ("reconstruct", "warmup")\n'
    "                else {**self.recon_cfg, **self.final_cfg}\n"
    "            )",
    "apply_phase mask_cfg",
)

at = replace_once(
    at,
    "        # Per-phase BKD curriculum (reconstruct: heavy annealed dropout;\n"
    "        # structure: off).  No-op unless a phase block sets batch_key_dropout.\n"
    "        self._apply_bkd_cfg(pl_module, phase)",
    "        # Per-phase BKD curriculum (reconstruct: heavy annealed dropout;\n"
    "        # structure: off).  No-op unless a phase block sets batch_key_dropout.\n"
    "        self._apply_bkd_cfg(pl_module, phase)\n"
    "\n"
    "        # BKD-coupled open gates (dedicated warmup phase only).\n"
    "        self._apply_open_gate_cfg(pl_module, phase)",
    "apply_phase open gate call",
)

# New method, inserted before _log_bkd_p
at = replace_once(
    at,
    "    def _log_bkd_p(self, pl_module: pl.LightningModule) -> None:",
    "    def _apply_open_gate_cfg(self, pl_module: pl.LightningModule, phase: str) -> None:\n"
    '        \"\"\"Toggle the BKD-coupled open-gate override (warmup phase only).\n'
    "\n"
    "        Active exactly when the phase is ``warmup`` and the warmup block sets\n"
    "        ``open_gate_bkd_coupled: true``.  The constant c_end is auto-measured\n"
    "        by the gated module on its first forward (the frozen learned-gate\n"
    "        init value), so the warmup -> structure switch is continuous.\n"
    '        \"\"\"\n'
    '        active = phase == "warmup" and bool(\n'
    '            self.warmup_cfg.get("open_gate_bkd_coupled", False)\n'
    "        )\n"
    "        n_mod = 0\n"
    "        for mod in pl_module.modules():\n"
    '            if hasattr(mod, "set_open_gate_mode"):\n'
    "                mod.set_open_gate_mode(active)\n"
    "                n_mod += 1\n"
    "        if active and n_mod == 0:\n"
    "            logger.warning(\n"
    '                "[adaptive] warmup.open_gate_bkd_coupled set but no gated "\n'
    '                "attention module (GatedSelfAttention) found."\n'
    "            )\n"
    "        if n_mod:\n"
    "            pl_module.log(\n"
    '                "open_gate_active", float(active), on_step=False, on_epoch=True\n'
    "            )\n"
    "\n"
    "    def _log_bkd_p(self, pl_module: pl.LightningModule) -> None:",
    "open-gate method",
)

AT.write_text(at, encoding="utf-8")
print("trainer edit 3c OK")
