"""One-off patch: header for the mkladder_film experiment variant."""
import io

p = ("experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/"
     "adaptive_nhsic_ladder_gt_ctx_mkladder_film/config.yaml")
s = io.open(p, encoding="utf-8").read()

old = (
    "# VARIANT of adaptive_nhsic_ladder_gt_ctx_film (same HSIC-as-CONSTRAINT Lagrangian\n"
)
# After the rename, the header still describes the mkladder variant; prepend
# the FiLM delta.  Anchor on the (renamed) VARIANT line.
anchor = (
    "# VARIANT of adaptive_nhsic_ladder_gt_ctx (same HSIC-as-CONSTRAINT Lagrangian\n"
)
assert s.count(anchor) == 1, "header anchor not found"
new_head = (
    "# VARIANT of adaptive_nhsic_ladder_gt_ctx_mkladder (same count-based\n"
    "# bkd_min_keys_ladder: [1..10], deterministic ON) with ONE change: the\n"
    "# adjacency context is injected via FiLM (per_node_adjacency_context:\n"
    "# film) instead of input concatenation.  A shared conditioner MLP maps\n"
    "# the detached applied-adjacency row to per-channel scale/shift of the\n"
    "# per-node decoder hidden activation (zero-init => identical to the\n"
    "# unconditioned decoder at step 0), so the context MULTIPLICATIVELY\n"
    "# switches decoder features per selected key subset instead of being a\n"
    "# washable additive input.  Motivation: at min_keys=1 the regressor must\n"
    "# fit a different one-key function per batch - a switch, not a feature.\n"
    "#\n"
    "# Parent-arm description follows:\n"
)
s = s.replace(anchor, new_head + anchor.replace(
    "# VARIANT of adaptive_nhsic_ladder_gt_ctx",
    "# VARIANT (parent arm) of adaptive_nhsic_ladder_gt_ctx"))
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("film config header patched")
