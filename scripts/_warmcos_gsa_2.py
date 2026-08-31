"""Edit script 2/2: set_open_gate_mode setter + open-gate forward branch in
GatedSelfAttention. Idempotent anchors."""
from pathlib import Path

GSA = Path("causaliT/core/modules/gated_self_attention.py")


def replace_once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: anchor found {text.count(old)}x"
    return text.replace(old, new)


gsa = GSA.read_text(encoding="utf-8")

gsa = replace_once(
    gsa,
    "    def set_bkd_phase_active(self, active: bool) -> None:\n"
    "        \"\"\"Enable/disable BKD application for the current training phase.\"\"\"\n"
    "        self._bkd_phase_active = bool(active)",
    "    def set_bkd_phase_active(self, active: bool) -> None:\n"
    "        \"\"\"Enable/disable BKD application for the current training phase.\"\"\"\n"
    "        self._bkd_phase_active = bool(active)\n"
    "\n"
    "    def set_open_gate_mode(\n"
    "        self, active: bool, c_end: Optional[float] = None\n"
    "    ) -> None:\n"
    "        \"\"\"Toggle the BKD-coupled open-gate override (warmup phase).\n"
    "\n"
    "        When active, forward() replaces the learned structure gate with the\n"
    "        constant ``c(t) = c_end + (1-c_end)*(p(t)-p_end)/(p_max-p_end)``\n"
    "        read live from the BKD schedule.  ``c_end=None`` auto-measures the\n"
    "        learned gate posterior (mean off-diagonal) on the first forward —\n"
    "        the gates are frozen all warmup, so this is exactly the value they\n"
    "        hold at the warmup -> structure switch (continuity by construction).\n"
    "        Deactivating restores the learned gates.\n"
    "        \"\"\"\n"
    "        self._open_gate_active = bool(active)\n"
    "        if c_end is not None:\n"
    "            self._open_gate_c_end = float(c_end)\n"
    "        if not active:\n"
    "            self.last_open_gate_c = None",
    "set_open_gate_mode",
)

gsa = replace_once(
    gsa,
    "            p_directed = torch.full_like(S_sym, c) * direction\n"
    "        else:",
    "            p_directed = torch.full_like(S_sym, c) * direction\n"
    "        elif self._open_gate_active:\n"
    "            # ---- BKD-coupled open gates (dedicated warmup phase) --------\n"
    "            p_now = self._current_bkd_p()\n"
    "            p_end = float(self._bkd_p1) if self._bkd_p1 is not None else 0.0\n"
    "            p_max = (\n"
    "                float(self._bkd_p_base) + float(self._bkd_amp or 0.0)\n"
    "                if self._bkd_p_base is not None\n"
    "                else None\n"
    "            )\n"
    "            if self._open_gate_c_end is None:\n"
    "                # Auto-measure c_end: the learned-gate eval posterior, mean\n"
    "                # over off-diagonal entries, at the CURRENT (init) structural\n"
    "                # state.  The gates are frozen for the whole warmup, so this\n"
    "                # is exactly the value they will hold at the switch.\n"
    "                with torch.no_grad():\n"
    "                    pi0 = torch.sigmoid(S_sym - self._l0_offset) * torch.sigmoid(\n"
    "                        A_anti / self.dir_beta + self.dir_bias\n"
    "                    )\n"
    "                    off = ~torch.eye(N, device=pi0.device, dtype=torch.bool)\n"
    "                    self._open_gate_c_end = float(pi0[:, off].mean())\n"
    "            if p_now is None or p_max is None or p_max <= p_end:\n"
    "                c = self._open_gate_c_end\n"
    "            else:\n"
    "                c = self._open_gate_c_end + (1.0 - self._open_gate_c_end) * (\n"
    "                    p_now - p_end\n"
    "                ) / (p_max - p_end)\n"
    "            c = float(min(max(c, 0.0), 1.0))\n"
    "            self.last_open_gate_c = c\n"
    "            structure = torch.full_like(S_sym, c)\n"
    "            p_edge_undirected = torch.full_like(S_sym, c)\n"
    "            direction = torch.full_like(S_sym, 0.5)\n"
    "            p_directed = torch.full_like(S_sym, c) * direction\n"
    "        else:",
    "open-gate branch",
)

GSA.write_text(gsa, encoding="utf-8")
print("edit script 2 applied OK")
