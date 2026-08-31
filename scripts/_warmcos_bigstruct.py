"""warmcos config: one big terminal structure phase (config-only)."""
from pathlib import Path

p = Path("experiments/6_INVESTIGATIONS/HSIC_OPT_2/"
         "bkd_warmup_06_global_nonorm_hsicbkd_d20_base_sgd_frozenbw_warmcos/config.yaml")
c = p.read_text(encoding="utf-8")

old = "  structure:\n    max_epochs: 300\n    min_epochs: 100"
new = (
    "  structure:\n"
    "    # ONE big terminal structure phase: min_epochs = the whole post-warmup\n"
    "    # budget (10000 - 800) suppresses BOTH early exits (drop + HSIC\n"
    "    # plateau), and max_epochs is lifted above it so the safety cap never\n"
    "    # fires: the phase runs to the run-level epoch budget.  No alternation\n"
    "    # back to reconstruct - the cosine-BKD warmup pre-trains a generic\n"
    "    # decoder, and re-fitting it after structure (co-adaptation) is exactly\n"
    "    # the failure mode this arm removes.\n"
    "    max_epochs: 10000\n"
    "    min_epochs: 9200"
)
assert c.count(old) == 1, f"anchor count {c.count(old)}"
c = c.replace(old, new)

old2 = "#   decoder).  Everything else identical to ..._sgd_frozenbw."
new2 = ("#   decoder).  The structure phase never exits (min_epochs=9200 spans\n"
        "#   the whole post-warmup budget): warmup -> ONE structure phase -> end.\n"
        "#   Everything else identical to ..._sgd_frozenbw.")
assert c.count(old2) == 1, f"header anchor count {c.count(old2)}"
c = c.replace(old2, new2)

p.write_text(c, encoding="utf-8")
print("config updated")
