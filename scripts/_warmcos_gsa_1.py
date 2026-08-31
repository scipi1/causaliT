"""Edit script 1/2: cosine BKD schedule + set_bkd_schedule kwargs in
GatedSelfAttention. Idempotent anchors."""
from pathlib import Path

GSA = Path("causaliT/core/modules/gated_self_attention.py")


def replace_once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: anchor found {text.count(old)}x"
    return text.replace(old, new)


gsa = GSA.read_text(encoding="utf-8")

gsa = replace_once(
    gsa,
    "        frac = min(1.0, float(self._bkd_step.item()) / float(self._bkd_anneal))\n"
    "        return float(self._bkd_p0) + frac * (float(self._bkd_p1) - float(self._bkd_p0))",
    "        frac = min(1.0, float(self._bkd_step.item()) / float(self._bkd_anneal))\n"
    "        if self._bkd_schedule == \"cosine\":\n"
    "            # Periodic warmup curriculum: oscillates in [p_base - amp,\n"
    "            # p_base + amp] under a linearly decaying envelope, landing\n"
    "            # exactly on p1 (p_end) at frac = 1.\n"
    "            p_end = float(self._bkd_p1)\n"
    "            p_base = (float(self._bkd_p_base) if self._bkd_p_base is not None\n"
    "                      else float(self._bkd_p0))\n"
    "            amp = float(self._bkd_amp) if self._bkd_amp is not None else 0.0\n"
    "            cycles = float(self._bkd_cycles) if self._bkd_cycles else 1.0\n"
    "            osc = 0.5 * (1.0 + math.cos(2.0 * math.pi * cycles * frac))\n"
    "            return p_end + (1.0 - frac) * ((p_base - p_end) + amp * osc)\n"
    "        return float(self._bkd_p0) + frac * (float(self._bkd_p1) - float(self._bkd_p0))",
    "cosine schedule",
)

gsa = replace_once(
    gsa,
    "        p1: Optional[float] = None,\n"
    "        annealing_batches: Optional[int] = None,\n"
    "    ) -> None:\n"
    "        \"\"\"Override the BKD schedule at run time (adaptive-trainer phase\n"
    "        controller).  The step counter is NOT reset: the anneal stays a\n"
    "        global, run-level clock.\"\"\"\n"
    "        self._bkd_p0 = p0\n"
    "        self._bkd_p1 = p1 if p1 is not None else p0\n"
    "        self._bkd_anneal = annealing_batches",
    "        p1: Optional[float] = None,\n"
    "        annealing_batches: Optional[int] = None,\n"
    "        schedule: str = \"linear\",\n"
    "        p_base: Optional[float] = None,\n"
    "        amp: Optional[float] = None,\n"
    "        cycles: Optional[float] = None,\n"
    "    ) -> None:\n"
    "        \"\"\"Override the BKD schedule at run time (adaptive-trainer phase\n"
    "        controller).  The step counter is NOT reset: the anneal stays a\n"
    "        global, run-level clock.\n"
    "\n"
    "        ``schedule=\"cosine\"`` selects the periodic warmup curriculum (see\n"
    "        ``_current_bkd_p``); ``p_base``/``amp``/``cycles`` are its\n"
    "        oscillation center / amplitude / period count and ``p1`` is the\n"
    "        landing value (p_end).\"\"\"\n"
    "        self._bkd_p0 = p0\n"
    "        self._bkd_p1 = p1 if p1 is not None else p0\n"
    "        self._bkd_anneal = annealing_batches\n"
    "        self._bkd_schedule = str(schedule)\n"
    "        self._bkd_p_base = p_base\n"
    "        self._bkd_amp = amp\n"
    "        self._bkd_cycles = cycles",
    "set_bkd_schedule",
)

GSA.write_text(gsa, encoding="utf-8")
print("edit script 1 applied OK")
