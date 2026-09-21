"""One-off patcher v2: extend scripts/_gen_hsic_constraint_arms.ps1 with
per-arm rhomax / topk / bkda / bkdx options and two new arms:
  lagrangian_l0_e5_topk9, lagrangian_l0_e5_bkd_anneal
Both set hsic_bkd_exclude_dropped: false (under attw_softmax the attention
weight IS the pair weight; a dropped causal key already gets ~0 weight
through its gate, so excluding the pair would remove the very signal that
should push the gate up).
"""
from pathlib import Path

p = Path("scripts/_gen_hsic_constraint_arms.ps1")
src = p.read_text()

old_tail = (
    "    @{ name = \"augmented_l0_e5\";         l0 = \"1.0e-5\"; kappa = \"0.0\";    dlr = \"1.0\"; rho = \"1.0\";\n"
    "       note = \"Augmented Lagrangian arm: rho_init=1.0 with NOTEARS-style x2 escalation.\" }\n"
    ")"
)
new_tail = (
    "    @{ name = \"augmented_l0_e5\";         l0 = \"1.0e-5\"; kappa = \"0.0\";    dlr = \"1.0\"; rho = \"1.0\";\n"
    "       note = \"Augmented Lagrangian arm: rho_init=1.0 with NOTEARS-style x2 escalation.\" },\n"
    "    @{ name = \"lagrangian_l0_e5_topk9\";      l0 = \"1.0e-5\"; kappa = \"0.0\"; dlr = \"1.0\"; rho = \"0.0\"; rhomax = \"1.0e6\";\n"
    "       topk = $true; bkdx = $false;\n"
    "       note = \"TOP-K arm: hard per-row source budget k=9 (true max in-degree), noisy_hard_k slack annealed to 0 over 50k batches; hsic_bkd_exclude_dropped=false (softmax pair weight already gates dropped keys).\" },\n"
    "    @{ name = \"lagrangian_l0_e5_bkd_anneal\"; l0 = \"1.0e-5\"; kappa = \"0.0\"; dlr = \"1.0\"; rho = \"0.0\"; rhomax = \"1.0e6\";\n"
    "       bkda = $true; bkdx = $false;\n"
    "       note = \"BKD-anneal arm: batch key dropout 0.8 -> 0 over 50k batches (stochastic symmetry breaking; best auroc_self retention of past campaigns); hsic_bkd_exclude_dropped=false.\" }\n"
    ")"
)
assert old_tail in src, "arms table tail not found"
src = src.replace(old_tail, new_tail)

old_loop = "foreach ($arm in $arms) {\n    $dir = Join-Path $base $arm.name"
new_loop = (
    "foreach ($arm in $arms) {\n"
    "    if (-not $arm.ContainsKey(\"rhomax\")) { $arm.rhomax = \"1.0e6\" }\n"
    "    if (-not $arm.ContainsKey(\"topk\"))   { $arm.topk = $false }\n"
    "    if (-not $arm.ContainsKey(\"bkda\"))   { $arm.bkda = $false }\n"
    "    if (-not $arm.ContainsKey(\"bkdx\"))   { $arm.bkdx = $true }\n"
    "    $dir = Join-Path $base $arm.name"
)
assert old_loop in src, "loop head not found"
src = src.replace(old_loop, new_loop)

old_rho = "$out.Add(\"    rho_max: 1.0e6\")"
assert old_rho in src, "rho_max line not found"
src = src.replace(old_rho, "$out.Add(\"    rho_max: $($arm.rhomax)\")")

old_else = "        } else {\n            $out.Add($line)"
branches = """        } elseif ($arm.topk -and $t -eq "  n_input: 18") {
            $out.Add($line)
            $out.Add("  # ---- TOP_K regressor budget (dense-regressor fix) ----")
            $out.Add("  topk_k: 9   # true max in-degree (dag_adj_mask.csv row sums)")
            $out.Add("  topk_method: noisy_hard_k")
            $out.Add("  topk_slack_init: null")
            $out.Add("  topk_slack_final: 0")
            $out.Add("  topk_annealing_batches: 50000   # = 25k optimizer steps (clock ticks 2x/step with cross-fit)")
            $out.Add("  topk_per_row_slack: true")
        } elseif ($arm.topk -and $t -eq "    per_node_output_hidden: `${experiment.per_node_output_hidden}") {
            $out.Add($line)
            $out.Add("    topk_k: `${experiment.topk_k}")
            $out.Add("    topk_method: `${experiment.topk_method}")
            $out.Add("    topk_slack_init: `${experiment.topk_slack_init}")
            $out.Add("    topk_slack_final: `${experiment.topk_slack_final}")
            $out.Add("    topk_annealing_batches: `${experiment.topk_annealing_batches}")
            $out.Add("    topk_per_row_slack: `${experiment.topk_per_row_slack}")
        } elseif ($arm.bkda -and $t -eq "    batch_key_dropout: 0.05") {
            $out.Add("    batch_key_dropout: 0.8")
        } elseif ($arm.bkda -and $t -eq "    batch_key_dropout_p_final: 0.05") {
            $out.Add("    batch_key_dropout_p_final: 0.0")
        } elseif ($arm.bkda -and $t -eq "    batch_key_dropout_annealing_batches: null") {
            $out.Add("    batch_key_dropout_annealing_batches: 50000")
        } elseif ((-not $arm.bkdx) -and $t -eq "  hsic_bkd_exclude_dropped: true") {
            $out.Add("  hsic_bkd_exclude_dropped: false  # attw_softmax: the gate weight already down-weights dropped keys")
        } else {
            $out.Add($line)"""
assert old_else in src, "else branch not found"
src = src.replace(old_else, branches)

p.write_text(src)
print("patched", p)
