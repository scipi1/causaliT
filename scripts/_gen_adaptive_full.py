from pathlib import Path

src_path = Path("experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/adaptive_nhsic_ladder_smoke/config.yaml")
src = src_path.read_text(encoding="utf-8")

for name, tol in (("adaptive_nhsic_ladder_gt", "0.11"), ("adaptive_nhsic_ladder_mid", "0.075")):
    s = src
    s = s.replace("# HSIC_CONSTRAINT / adaptive_nhsic_ladder_smoke",
                  f"# HSIC_CONSTRAINT / {name}")
    s = s.replace("python -m causaliT.cli train --exp_id 6_INVESTIGATIONS/HSIC_CONSTRAINT/adaptive_nhsic_ladder_smoke",
                  f"python -m causaliT.cli train --exp_id 6_INVESTIGATIONS/HSIC_CONSTRAINT/{name}")
    # Back to the full 20-node benchmark
    s = s.replace("dataset: scm2_continuous",
                  "dataset: random_n20_k4_er_nonlinear_gaussian_s1")
    s = s.replace("train_file: ds_train.npz", "train_file: null")
    s = s.replace("test_file: ds_test.npz", "test_file: null")
    s = s.replace("n_nodes: 10", "n_nodes: 20")
    s = s.replace("n_source: 5", "n_source: 2")
    s = s.replace("n_input: 5", "n_input: 18")
    # Budget: warmup 300 + cycles; 4000 total
    s = s.replace("max_epochs: 800", "max_epochs: 4000")
    s = s.replace("total_epoch_budget: 800", "total_epoch_budget: 4000")
    s = s.replace("    max_epochs: 150", "    max_epochs: 300")
    # Tolerance per arm
    s = s.replace("tolerance: 0.075", f"tolerance: {tol}")
    out = Path(f"experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/{name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(s, encoding="utf-8")
    print("wrote", out / "config.yaml")

import yaml
for name, tol in (("adaptive_nhsic_ladder_gt", 0.11), ("adaptive_nhsic_ladder_mid", 0.075)):
    c = yaml.safe_load(open(f"experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/{name}/config.yaml"))
    t = c["training"]; ad = c["adaptive_training"]
    assert c["data"]["dataset"] == "random_n20_k4_er_nonlinear_gaussian_s1"
    assert t["hsic_constraint"]["tolerance"] == tol and t["hsic_mode"] == "normalized"
    assert t["lambda_hsic"] == 0.0 and t["gradient_surgery"] is False
    assert t["hsic_aggregation"] == "attw_softmax" and t["lambda_l0"] == 1e-05
    assert ad["bkd_ladder"] == [0.8, 0.6, 0.4, 0.2, 0.0] and ad["start_phase"] == "warmup"
    assert ad["total_epoch_budget"] == 4000 and ad["warmup"]["max_epochs"] == 300
    assert "lambda_hsic" not in ad["structure"]
print("BOTH ARMS OK")
