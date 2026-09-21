"""One-off patch: header for the mkladder_bigmlp_film experiment variant."""
import io

p = ("experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/"
     "adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film/config.yaml")
s = io.open(p, encoding="utf-8").read()

anchor = (
    "# VARIANT of adaptive_nhsic_ladder_gt_ctx_mkladder (same count-based\n"
)
assert s.count(anchor) == 1, "header anchor not found"
new_head = (
    "# VARIANT of adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp (same count\n"
    "# ladder, same doubled decoder capacity) with ONE change: the adjacency\n"
    "# context is injected via FiLM (per_node_adjacency_context: film) instead\n"
    "# of input concatenation.  A shared conditioner MLP maps the detached\n"
    "# applied-adjacency row to per-channel scale/shift of each per-node\n"
    "# decoder hidden activation (zero-init => identical to the unconditioned\n"
    "# decoder at step 0), so the context MULTIPLICATIVELY switches decoder\n"
    "# features per selected key subset rather than being a washable additive\n"
    "# input.  Combines both fixes for the min_keys=1 warmup regime: capacity\n"
    "# (bigmlp) + switch-style conditioning (film).\n"
    "#\n"
    "# FALSIFIABLE PREDICTION vs _bigmlp: lower warmup residual floor and\n"
    "# faster val_x_mae descent at rungs 1-3 (the conditioner can hard-switch\n"
    "# features by key subset instead of learning it from concat); converges\n"
    "# to the same curves once min_keys is large.\n"
    "#\n"
    "# Parent-arm description follows:\n"
)
s = s.replace(
    anchor,
    new_head + anchor.replace(
        "# VARIANT of adaptive_nhsic_ladder_gt_ctx_mkladder",
        "# VARIANT (parent arm) of adaptive_nhsic_ladder_gt_ctx_mkladder",
    ),
)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("bigmlp_film header patched")
