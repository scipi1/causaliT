"""Part 2: training hygiene (final-metrics table + train/val curves + NOTEARS/L0)."""
import nbformat as nbf

md = (
    "## 1. Training hygiene\n"
    "\n"
    "Pre-requisite before any structural analysis: the fit must be good. **R2 is the\n"
    "primary indicator** — check it first (also the macro/src variants: `r2_src`\n"
    "covers the hard-to-fit source nodes). Final values are the last logged epoch\n"
    "(plain run: no summary JSON)."
)

c1 = (
    "# ---- Final-metrics table (last logged epoch; R2 first) ----------------------\n"
    "KEYS = [\"val_x_r2\", \"test_x_r2\", \"val_x_r2_macro\", \"val_x_r2_src\", \"val_x_r2_endo\",\n"
    "        \"val_x_mae\", \"test_x_mae\", \"val_x_rmse\", \"val_hsic\", \"test_hsic\",\n"
    "        \"val_notears\", \"val_kappa_eff\", \"val_l0_penalty\",\n"
    "        \"val_hsic_desc_kept_frac\", \"val_loss\"]\n"
    "tab = pd.Series({k: last_value(k) for k in KEYS}).dropna()\n"
    "print(tab.to_string(float_format=lambda v: f\"{v:.4g}\"))"
)

c2 = (
    "# ---- Train/val curves ---------------------------------------------------------\n"
    "PANELS = [(\"loss_x\", \"MSE loss [-]\", True, None), (\"x_mae\", \"MAE [-]\", False, None),\n"
    "          (\"x_r2\", \"R2 [-]\", False, (0, 1)), (\"hsic\", \"HSIC [-]\", False, None)]\n"
    "\n"
    "fig, axes = plt.subplots(1, len(PANELS), figsize=(3.6 * len(PANELS), 3.4))\n"
    "for ax, (stem, ylab, logy, ylim) in zip(axes, PANELS):\n"
    "    for split, col in [(\"train\", OKABE_ITO[0]), (\"val\", OKABE_ITO[3])]:\n"
    "        c = f\"{split}_{stem}\"\n"
    "        if c in metrics:\n"
    "            e, v = metric_series(c)\n"
    "            ax.plot(e, v, label=split, color=col, lw=1.2)\n"
    "    if logy:\n"
    "        ax.set_yscale(\"log\")\n"
    "    if ylim is not None:\n"
    "        ax.set_ylim(ylim)\n"
    "    ax.set_ylabel(ylab)\n"
    "    ax.set_title(stem)\n"
    "    ax.set_xlim(0, E_MAX)\n"
    "    ax.set_xlabel(\"epoch [-]\")\n"
    "    ax.legend(fontsize=8)\n"
    "fig.suptitle(\"Train/val metrics\", fontweight=\"bold\")\n"
    "fig.tight_layout()\n"
    "save_fig(fig, \"train_val_curves\")\n"
    "plt.show()"
)

c3 = (
    "# ---- NOTEARS / L0 / descendant-mask activity (still logged by the forecaster) ----\n"
    "fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))\n"
    "\n"
    "if \"val_notears\" in metrics:\n"
    "    e, v = metric_series(\"val_notears\")\n"
    "    axes[0].plot(e, v, color=OKABE_ITO[0], lw=1.2, label=\"val NOTEARS h(A)\")\n"
    "if \"val_l0_penalty\" in metrics:\n"
    "    axt = axes[0].twinx()\n"
    "    e, v = metric_series(\"val_l0_penalty\")\n"
    "    axt.plot(e, v, color=OKABE_ITO[3], lw=1.0, alpha=0.7, label=\"val L0 [edges]\")\n"
    "    axt.set_ylabel(\"L0 penalty [edges]\", color=OKABE_ITO[3])\n"
    "    axt.tick_params(axis=\"y\", colors=OKABE_ITO[3])\n"
    "axes[0].set_ylabel(\"NOTEARS h(A) [-]\")\n"
    "axes[0].set_title(\"NOTEARS and L0\")\n"
    "\n"
    "for col, lbl, c in [(\"train_hsic_desc_kept_frac\", \"train\", OKABE_ITO[0]),\n"
    "                    (\"val_hsic_desc_kept_frac\", \"val\", OKABE_ITO[3])]:\n"
    "    if col in metrics:\n"
    "        e, v = metric_series(col)\n"
    "        axes[1].plot(e, v, label=lbl, color=c, lw=1.1)\n"
    "axes[1].set_ylim(0, 1.02)\n"
    "axes[1].set_ylabel(\"kept fraction [-]\")\n"
    "axes[1].set_title(\"Descendant-mask activity\")\n"
    "axes[1].legend(fontsize=8)\n"
    "\n"
    "for ax in axes:\n"
    "    ax.set_xlim(0, E_MAX)\n"
    "    ax.set_xlabel(\"epoch [-]\")\n"
    "fig.tight_layout()\n"
    "save_fig(fig, \"regularizer_activity\")\n"
    "plt.show()"
)

cells = [nbf.v4.new_markdown_cell(md)] + [nbf.v4.new_code_cell(c) for c in (c1, c2, c3)]
nbf.write(nbf.v4.new_notebook(cells=cells), "scripts/_nb_part2.ipynb")
print("part2 ok")
