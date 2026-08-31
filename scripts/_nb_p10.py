"""Part 10: summary cell."""
import nbformat as nbf

md = (
    "## 7. Summary\n"
    "\n"
    "Key numbers and the verdict: R2 gate, final alignment quality, the\n"
    "gradient-field readout (parent alignment and radial share of the evidence),\n"
    "movement signature, and the Q-vs-M attribution split."
)

c1 = (
    "# ---- Key numbers and verdict --------------------------------------------------------------\n"
    "rows = {\n"
    "    \"val R2\": last_value(\"val_x_r2\"),\n"
    "    \"test R2\": last_value(\"test_x_r2\"),\n"
    "    \"val R2 macro\": last_value(\"val_x_r2_macro\"),\n"
    "    \"val R2 src\": last_value(\"val_x_r2_src\"),\n"
    "    \"val MAE\": last_value(\"val_x_mae\"),\n"
    "    \"val HSIC\": last_value(\"val_hsic\"),\n"
    "    \"val NOTEARS\": last_value(\"val_notears\"),\n"
    "    \"val L0 [edges]\": last_value(\"val_l0_penalty\"),\n"
    "    \"final SHD\": int(SHD_ALL[-1, 0]),\n"
    "    \"best SHD\": f\"{int(SHD_ALL[:, 0].min())} @ epoch \"\n"
    "                f\"{EPOCHS[int(SHD_ALL[:, 0].argmin())]}\",\n"
    "    \"SHD attribution (final full / Q-only / M-only)\":\n"
    "        f\"{SHD_ALL[-1, 0]} / {SHD_qonly[-1]} / {SHD_monly[-1]}\",\n"
    "    \"centroid align (final mean, X)\": np.nanmean(ALIGN[-1, N_S:]),\n"
    "    \"centroid align (final min, X)\": np.nanmin(ALIGN[-1, N_S:]),\n"
    "    \"gradient-parent align (first->last, mean X)\":\n"
    "        f\"{np.nanmean(G_PAR[0, N_S:]):+.3f} -> {np.nanmean(G_PAR[-1, N_S:]):+.3f}\",\n"
    "    \"gradient radial share (first->last)\":\n"
    "        f\"{G_RAD[0].mean():.2f} -> {G_RAD[-1].mean():.2f}\",\n"
    "}\n"
    "print(pd.Series(rows).to_string())\n"
    "print()\n"
    "print(f\"most moved: {[NODES[i] for i in order_mv[:5]]}\")\n"
    "conv = np.isfinite(ALIGN[-1]) & HAS_PAR\n"
    "print(f\"nodes with parents converged (final cos > 0.9): \"\n"
    "      f\"{int((ALIGN[-1, conv] > 0.9).sum())}/{int(conv.sum())}\")"
)

cells = [nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(c1)]
nbf.write(nbf.v4.new_notebook(cells=cells), "scripts/_nb_part10.ipynb")
print("part10 ok")
