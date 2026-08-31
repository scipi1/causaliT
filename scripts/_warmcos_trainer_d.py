"""Edit script 3d: warmup dispatch in on_validation_epoch_end + split-key map."""
from pathlib import Path

AT = Path("causaliT/training/adaptive_trainer.py")


def replace_once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: anchor found {text.count(old)}x"
    return text.replace(old, new)


at = AT.read_text(encoding="utf-8")

# Warmup dispatch: fixed budget, then straight to structure.  Placed after the
# final_reconstruct entry trigger, before the reconstruct dispatch.
at = replace_once(
    at,
    "        # ---------------- Reconstruct phase: plateau / budget ----------------\n"
    '        if self.current_phase == "reconstruct":',
    '        # ---------- Warmup phase: fixed budget -> structure ----------------\n'
    "        # The cosine BKD curriculum must run to completion, so the ONLY exit\n"
    "        # is the epoch budget (no plateau early-exit).\n"
    '        if self.current_phase == "warmup":\n'
    "            for mod in pl_module.modules():\n"
    '                if getattr(mod, "last_open_gate_c", None) is not None:\n'
    '                    pl_module.log("open_gate_c", float(mod.last_open_gate_c),\n'
    "                                  on_step=False, on_epoch=True)\n"
    "                    break\n"
    "            if phase_epochs >= self.warmup_max_epochs:\n"
    "                self._record_transition(\n"
    '                    trainer, pl_module, "warmup_budget",\n'
    '                    from_phase="warmup", to_phase="structure",\n'
    "                    monitor_val=current,\n"
    "                )\n"
    "                self._phase_index += 1\n"
    '                self._apply_phase(trainer, pl_module, "structure")\n'
    "            return\n"
    "\n"
    "        # ---------------- Reconstruct phase: plateau / budget ----------------\n"
    '        if self.current_phase == "reconstruct":',
    "warmup dispatch",
)

# Warmup trains on the reconstruct split (cross-fitting).
at = replace_once(
    at,
    "        training set) is never swapped.\n"
    '        """\n'
    "        if (\n"
    "            self.swap_splits",
    "        training set) is never swapped.  ``warmup`` always uses the\n"
    "        reconstruct subset (and is never swapped: it runs before cycle 1).\n"
    '        """\n'
    '        if phase == "warmup":\n'
    '            return "reconstruct"\n'
    "        if (\n"
    "            self.swap_splits",
    "split key",
)

AT.write_text(at, encoding="utf-8")
print("trainer edit 3d OK")
