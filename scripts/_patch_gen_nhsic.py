from pathlib import Path
p = Path("scripts/_gen_hsic_constraint_arms.ps1")
src = p.read_text(encoding="utf-8")

# 1) new nHSIC arms (tolerance from scripts/calibrate_hsic_tolerance.py)
old_tail = """       note = "ORACLE VALIDATION: expected-SHD constraint + BKD 0.8 -> 0 anneal." }
)"""
new_tail = """       note = "ORACLE VALIDATION: expected-SHD constraint + BKD 0.8 -> 0 anneal." },
    @{ name = "lagrangian_l0_e5_nhsic_null"; l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.043";
       note = "nHSIC arm, tolerance = permutation-null q99 (0.043): strict independence test; expected INFEASIBLE at current fit level (GT reads ~0.11)." },
    @{ name = "lagrangian_l0_e5_nhsic_mid";  l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.075";
       note = "nHSIC arm, tolerance = midpoint 0.075 between null q99 and GT level." },
    @{ name = "lagrangian_l0_e5_nhsic_gt";   l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.11";
       note = "nHSIC arm, tolerance = GT level (0.11): feasibility should hold near the truth; lam should plateau, auroc hold." }
)"""
assert old_tail in src, "arms tail"
src = src.replace(old_tail, new_tail)

# 2) defaults
old = '    if (-not $arm.ContainsKey("source")) { $arm.source = "hsic" }'
new = old + "\n" + '    if (-not $arm.ContainsKey("mode"))   { $arm.mode = "biased" }' + "\n" + '    if (-not $arm.ContainsKey("tol"))    { $arm.tol = "0.0" }'
assert old in src, "defaults"
src = src.replace(old, new)

# 3) emit tolerance per arm and hsic_mode override
old = '            $out.Add("    tolerance: 0.0")'
new = '            $out.Add("    tolerance: $($arm.tol)")'
assert old in src, "tol emit"
src = src.replace(old, new)

old = '''        } elseif ($t -eq "  hsic_mode: biased") {
            $out.Add($line)'''
if old not in src:
    # hsic_mode line currently falls through the else; add a branch before else
    old_else = "        } else {\n            $out.Add($line)"
    branch = '''        } elseif ($t -eq "  hsic_mode: biased") {
            $out.Add("  hsic_mode: $($arm.mode)")
        } else {
            $out.Add($line)'''
    assert old_else in src, "else branch"
    src = src.replace(old_else, branch)

p.write_text(src, encoding="utf-8")
print("patched generator")
