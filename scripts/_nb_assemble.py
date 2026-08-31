"""Assemble the simplified plain-trainer eval notebook from the parts."""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

SCRIPTS = [
    "scripts/_nb_p1.py", "scripts/_nb_p2.py", "scripts/_nb_p3.py",
    "scripts/_nb_p4.py", "scripts/_nb_p5.py", "scripts/_nb_p6.py",
    "scripts/_nb_p7.py", "scripts/_nb_p8.py", "scripts/_nb_p9.py",
    "scripts/_nb_p10.py",
]
PARTS = [f"scripts/_nb_part{i}.ipynb" for i in range(1, 11)]

for s in SCRIPTS:
    subprocess.run([sys.executable, s], check=True)

cells = []
for p in PARTS:
    cells.extend(nbf.read(p, as_version=4).cells)

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "venv", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

dst = Path("experiments/6_INVESTIGATIONS/HSIC_OPT_2/results/"
           "bkd_warmup_06_global_nonorm_hsicbkd_d20_base_sgd_frozenbw_joint_12189935/"
           "evaluate_updates.ipynb")
nbf.write(nb, dst)
nbf.validate(nb)
print(f"wrote {dst} with {len(cells)} cells (validated)")
