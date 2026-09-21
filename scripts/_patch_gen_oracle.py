from pathlib import Path
p = Path("scripts/_gen_hsic_constraint_arms.ps1")
src = p.read_text(encoding="utf-8")

# 1) new arms
old_tail = """       note = "BKD-anneal arm: batch key dropout 0.8 -> 0 over 50k batches (stochastic symmetry breaking; best auroc_self retention of past campaigns); hsic_bkd_exclude_dropped=false." }
)"""
new_tail = """       note = "BKD-anneal arm: batch key dropout 0.8 -> 0 over 50k batches (stochastic symmetry breaking; best auroc_self retention of past campaigns); hsic_bkd_exclude_dropped=false." },
    @{ name = "oracle_shd_dense";      l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; source = "oracle_shd";
       note = "ORACLE VALIDATION: constraint = expected SHD to the GT DAG (perfect-estimator simulation), dense regressor." },
    @{ name = "oracle_shd_topk9";      l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; source = "oracle_shd"; topk = $true; bkdx = $false;
       note = "ORACLE VALIDATION: expected-SHD constraint + k=9 top-k budget." },
    @{ name = "oracle_shd_bkd_anneal"; l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; source = "oracle_shd"; bkda = $true; bkdx = $false;
       note = "ORACLE VALIDATION: expected-SHD constraint + BKD 0.8 -> 0 anneal." }
)"""
assert old_tail in src, "arms tail"
src = src.replace(old_tail, new_tail)

# 2) default source
old = '    if (-not $arm.ContainsKey("bkdx"))   { $arm.bkdx = $true }'
new = old + "`n" + '    if (-not $arm.ContainsKey("source")) { $arm.source = "hsic" }'
assert old in src, "defaults"
src = src.replace(old, new)

# 3) emit source in the constraint block
old = '            $out.Add("    enabled: true")'
new = old + "`n" + '            $out.Add("    source: $($arm.source)")'
assert old in src, "source emit"
src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("patched generator")
