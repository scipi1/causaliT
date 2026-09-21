from pathlib import Path
p = Path("scripts/_gen_hsic_constraint_arms.ps1")
src = p.read_text(encoding="utf-8")

old_tail = """       note = "nHSIC arm, tolerance = GT level (0.11): feasibility should hold near the truth; lam should plateau, auroc hold." }
)"""
new_tail = """       note = "nHSIC arm, tolerance = GT level (0.11): feasibility should hold near the truth; lam should plateau, auroc hold." },
    @{ name = "lagrangian_l0_e5_nhsic_gt_bkd_anneal";  l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.11"; bkda = $true; bkdx = $false;
       note = "PRIMARY overnight arm: nHSIC + tol=GT level (0.11) + BKD 0.8 -> 0 anneal (best auroc-retention mechanism of past campaigns); hsic_bkd_exclude_dropped=false." },
    @{ name = "lagrangian_l0_e5_nhsic_mid_bkd_anneal"; l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.075"; bkda = $true; bkdx = $false;
       note = "Tolerance-boundary arm under BKD anneal: tol=0.075 midpoint; tests whether the feasible window survives symmetry breaking." },
    @{ name = "lagrangian_l0_e5_nhsic_gt_topk9";       l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.11"; topk = $true; bkdx = $false;
       note = "Deterministic counterpart: nHSIC + tol=GT level (0.11) + k=9 top-k budget; clean deterministic-vs-stochastic contrast vs the BKD-anneal arm." }
)"""
assert old_tail in src, "arms tail"
src = src.replace(old_tail, new_tail)
p.write_text(src, encoding="utf-8")
print("patched generator")
