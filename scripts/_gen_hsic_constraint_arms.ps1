# Generates the HSIC_CONSTRAINT experiment arms from the HSIC_OPT_3
# joint_mse_attw_softmax_l0 template (gradient routing + HSIC cross-fitting +
# attw_softmax aggregation, 20-node homogeneous ER).
#
# Each arm: same model/data/experiment sections; the training section is
# rewritten to the constraint regime:
#   lambda_hsic: 0.0, gradient_surgery: false, *_max_hsic_pct: 0.0,
#   log_l0_hsic_interference: false, + training.hsic_constraint block.
#
# Run from the repo root:  powershell -File scripts/_gen_hsic_constraint_arms.ps1

$ErrorActionPreference = "Stop"

$template = "experiments/6_INVESTIGATIONS/HSIC_OPT_3/joint_mse_attw_softmax_l0/config_joint_mse_attw_softmax_l0.yaml"
$base = "experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT"

$arms = @(
    @{ name = "lagrangian_l0_e5";        l0 = "1.0e-5"; kappa = "0.0";    dlr = "1.0"; rho = "0.0";
       note = "PRIMARY: pure Lagrangian (rho=0), L0 primal at 1e-5, dual_lr 1.0." },
    @{ name = "lagrangian_l0_e4";        l0 = "1.0e-4"; kappa = "0.0";    dlr = "1.0"; rho = "0.0";
       note = "L0 magnitude arm: 1e-4 (stronger sparsity pressure)." },
    @{ name = "lagrangian_l0_notears";   l0 = "1.0e-5"; kappa = "1.0e-2"; dlr = "1.0"; rho = "0.0";
       note = "NOTEARS arm: adds kappa=1e-2 acyclicity to the primal (template had it vacuous)." },
    @{ name = "lagrangian_l0_e5_dual03"; l0 = "1.0e-5"; kappa = "0.0";    dlr = "0.3"; rho = "0.0";
       note = "dual_lr sweep arm: slow ascent 0.3." },
    @{ name = "lagrangian_l0_e5_dual3";  l0 = "1.0e-5"; kappa = "0.0";    dlr = "3.0"; rho = "0.0";
       note = "dual_lr sweep arm: fast ascent 3.0." },
    @{ name = "augmented_l0_e5";         l0 = "1.0e-5"; kappa = "0.0";    dlr = "1.0"; rho = "1.0";
       note = "Augmented Lagrangian arm: rho_init=1.0 with NOTEARS-style x2 escalation." },
    @{ name = "lagrangian_l0_e5_topk9";      l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; rhomax = "1.0e6";
       topk = $true; bkdx = $false;
       note = "TOP-K arm: hard per-row source budget k=9 (true max in-degree), noisy_hard_k slack annealed to 0 over 50k batches; hsic_bkd_exclude_dropped=false (softmax pair weight already gates dropped keys)." },
    @{ name = "lagrangian_l0_e5_bkd_anneal"; l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; rhomax = "1.0e6";
       bkda = $true; bkdx = $false;
       note = "BKD-anneal arm: batch key dropout 0.8 -> 0 over 50k batches (stochastic symmetry breaking; best auroc_self retention of past campaigns); hsic_bkd_exclude_dropped=false." },
    @{ name = "oracle_shd_dense";      l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; source = "oracle_shd";
       note = "ORACLE VALIDATION: constraint = expected SHD to the GT DAG (perfect-estimator simulation), dense regressor." },
    @{ name = "oracle_shd_topk9";      l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; source = "oracle_shd"; topk = $true; bkdx = $false;
       note = "ORACLE VALIDATION: expected-SHD constraint + k=9 top-k budget." },
    @{ name = "oracle_shd_bkd_anneal"; l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; source = "oracle_shd"; bkda = $true; bkdx = $false;
       note = "ORACLE VALIDATION: expected-SHD constraint + BKD 0.8 -> 0 anneal." },
    @{ name = "lagrangian_l0_e5_nhsic_null"; l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.043";
       note = "nHSIC arm, tolerance = permutation-null q99 (0.043): strict independence test; expected INFEASIBLE at current fit level (GT reads ~0.11)." },
    @{ name = "lagrangian_l0_e5_nhsic_mid";  l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.075";
       note = "nHSIC arm, tolerance = midpoint 0.075 between null q99 and GT level." },
    @{ name = "lagrangian_l0_e5_nhsic_gt";   l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.11";
       note = "nHSIC arm, tolerance = GT level (0.11): feasibility should hold near the truth; lam should plateau, auroc hold." },
    @{ name = "lagrangian_l0_e5_nhsic_gt_bkd_anneal";  l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.11"; bkda = $true; bkdx = $false;
       note = "PRIMARY overnight arm: nHSIC + tol=GT level (0.11) + BKD 0.8 -> 0 anneal (best auroc-retention mechanism of past campaigns); hsic_bkd_exclude_dropped=false." },
    @{ name = "lagrangian_l0_e5_nhsic_mid_bkd_anneal"; l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.075"; bkda = $true; bkdx = $false;
       note = "Tolerance-boundary arm under BKD anneal: tol=0.075 midpoint; tests whether the feasible window survives symmetry breaking." },
    @{ name = "lagrangian_l0_e5_nhsic_gt_topk9";       l0 = "1.0e-5"; kappa = "0.0"; dlr = "1.0"; rho = "0.0"; mode = "normalized"; tol = "0.11"; topk = $true; bkdx = $false;
       note = "Deterministic counterpart: nHSIC + tol=GT level (0.11) + k=9 top-k budget; clean deterministic-vs-stochastic contrast vs the BKD-anneal arm." }
)

foreach ($arm in $arms) {
    if (-not $arm.ContainsKey("rhomax")) { $arm.rhomax = "1.0e6" }
    if (-not $arm.ContainsKey("topk"))   { $arm.topk = $false }
    if (-not $arm.ContainsKey("bkda"))   { $arm.bkda = $false }
    if (-not $arm.ContainsKey("bkdx"))   { $arm.bkdx = $true }
    if (-not $arm.ContainsKey("source")) { $arm.source = "hsic" }
    if (-not $arm.ContainsKey("mode"))   { $arm.mode = "biased" }
    if (-not $arm.ContainsKey("tol"))    { $arm.tol = "0.0" }
    $dir = Join-Path $base $arm.name
    New-Item -ItemType Directory -Force $dir | Out-Null

    $header = @"
# ===========================================================================
# HSIC_CONSTRAINT / $($arm.name)
# ===========================================================================
# First test of HSIC-as-CONSTRAINT (Lagrangian / augmented Lagrangian).
# The structural stream's PRIMAL objective is L0(+NOTEARS); HSIC
# (softmax-attention-weighted, cross-fitted) enters only as the monitored
# constraint:
#   L = L0 + kappa*NOTEARS + lam*(HSIC - eps) + (rho/2)*relu(HSIC - eps)^2
# with per-epoch dual ascent lam <- clip(lam + dual_lr*(EMA(HSIC) - eps),
# 0, dual_max) and NOTEARS-style rho escalation (x2 on stalled violation).
#
# THIS ARM: $($arm.note)
#   lambda_l0=$($arm.l0), kappa=$($arm.kappa), dual_lr=$($arm.dlr), rho_init=$($arm.rho)
#
# Base template: HSIC_OPT_3/joint_mse_attw_softmax_l0 (gradient routing +
# HSIC cross-fitting + attw_softmax aggregation, homogeneous 20-node ER).
# DISABLED vs template (the forecaster hard-errors on these combinations;
# the constraint REVERSES the HSIC vs L0/NOTEARS relationship):
#   gradient_surgery: false, lambda_hsic: 0.0, *_max_hsic_pct: 0.0.
#
# FALSIFIABLE PREDICTION: hsic/dual_lambda rises while dag/auroc_self is
# mis-ranked, then plateaus once the constraint is met; the ancestor-tilt
# of the parent arm should disappear because the HSIC pressure no longer
# expires with the gradient signal.  If lam hits dual_max while auroc_self
# decays, constraint and discovery have decoupled -> hypothesis dead.
#
# WATCH (metrics.csv): hsic/dual_lambda, hsic/constraint_violation,
#   dag/auroc_self, dag/parent_vs_ancestor_contrast_self, val_x_mae,
#   val_hsic (all-sources reference = held-out constraint check).
#
#   python -m causaliT.cli train --exp_id 6_INVESTIGATIONS/HSIC_CONSTRAINT/$($arm.name)
# ===========================================================================
"@

    $lines = Get-Content $template
    $out = New-Object System.Collections.Generic.List[string]
    $started = $false
    foreach ($line in $lines) {
        if (-not $started) {
            # Drop the entire template comment header; emit ours instead.
            if ($line -eq "experiment:") {
                $out.Add($header)
                $out.Add($line)
                $started = $true
            }
            continue
        }
        $t = $line.TrimEnd()
        if ($t -eq "  lambda_hsic: 1.0") {
            $out.Add("  lambda_hsic: 0.0  # OFF: replaced by hsic_constraint below")
        } elseif ($t -eq "  lambda_l0: 1.0") {
            $out.Add("  lambda_l0: $($arm.l0)")
        } elseif ($t -eq "  kappa: 0.0") {
            $out.Add("  kappa: $($arm.kappa)")
        } elseif ($t -eq "  kappa_max_hsic_pct: 0.1") {
            $out.Add("  kappa_max_hsic_pct: 0.0  # OFF: constraint mode reverses HSIC/L0")
        } elseif ($t.StartsWith("  lambda_l0_max_hsic_pct:")) {
            $out.Add("  lambda_l0_max_hsic_pct: 0.0  # OFF: constraint mode reverses HSIC/L0")
        } elseif ($t -eq "  log_l0_hsic_interference: true") {
            $out.Add("  log_l0_hsic_interference: false")
        } elseif ($t -eq "  gradient_surgery: true") {
            $out.Add("  gradient_surgery: false  # OFF: reversed relationship (see header)")
            $out.Add("  # THE ARM: HSIC as a Lagrangian constraint on the structural")
            $out.Add("  # stream (L0 + NOTEARS primal).  lam grows while the EMA of")
            $out.Add("  # the raw train HSIC exceeds tolerance, so the pressure does")
            $out.Add("  # not vanish with the HSIC gradient signal.")
            $out.Add("  hsic_constraint:")
            $out.Add("    enabled: true")
            $out.Add("    source: $($arm.source)")
            $out.Add("    tolerance: $($arm.tol)")
            $out.Add("    dual_init: 0.0")
            $out.Add("    dual_lr: $($arm.dlr)")
            $out.Add("    dual_max: 1000.0")
            $out.Add("    rho_init: $($arm.rho)")
            $out.Add("    rho_mult: 2.0")
            $out.Add("    rho_max: $($arm.rhomax)")
            $out.Add("    ema: 0.9")
        } elseif ($arm.topk -and $t -eq "  n_input: 18") {
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
        } elseif ($t -eq "  hsic_mode: biased") {
            $out.Add("  hsic_mode: $($arm.mode)")
        } else {
            $out.Add($line)
        }
    }

    if (-not $started) { throw "Template header terminator 'experiment:' not found." }
    $path = Join-Path $dir "config.yaml"
    Set-Content -Path $path -Value $out
    Write-Output "wrote $path"
}

