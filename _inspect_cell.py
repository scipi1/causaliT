import json

nb = json.load(open(
    r"experiments/6_INVESTIGATIONS/LARGER_DAGS/analyze_no_NT_hsic_signal.ipynb",
    encoding="utf-8"))
cell = [c for c in nb["cells"] if c.get("id") == "bkd_sweep_probe"][0]
src = "".join(cell["source"])
i = src.find("sub = bkd_dilution")
print(src[i:])
