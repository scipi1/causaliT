"""Patch scm_ds/scm.py: raw-value capture + sortability.json export in generate_ds."""
from pathlib import Path

P = Path("scm_ds/scm.py")
t = P.read_text(encoding="utf-8")


def replace_once(old, new, label):
    global t
    assert t.count(old) == 1, f"{label}: anchor found {t.count(old)}x"
    t = t.replace(old, new)


# 1) Capture raw (pre-normalization) values for the sortability analytics.
replace_once(
    "        # --------------------- Normalization -----------------------\n",
    "        # Raw (pre-normalization) value copies for the sortability\n"
    "        # analytics exported below: var-sortability on RAW values\n"
    "        # quantifies the variance staircase normalization may remove.\n"
    "        raw_input_np = input_np.copy() if input_np is not None else None\n"
    "        raw_source_np = source_np.copy() if source_np is not None else None\n"
    "\n"
    "        # --------------------- Normalization -----------------------\n",
    "raw-capture",
)

# 2) After normalization.json export: compute + export sortability.json.
replace_once(
    "        # Export dataset metadata for evaluation functions (NEW)\n",
    """        # ------------- sortability analytics (Reisach et al. 2021/2023) -----
        # Path-weighted order alignment (CausalDisco convention).  Computed on
        # the STORED (normalized) values -- the matrix the model trains on --
        # and on the RAW pre-normalization values.  Failures here must never
        # break dataset generation, hence the guard.
        try:
            from scm_ds.sortability import var_sortability, r2_sortability

            avail_labels = []
            if self.source_labels is not None:
                avail_labels += list(self.source_labels)
            avail_labels += list(self.input_labels)

            def _values_matrix(src, inp):
                # columns aligned with ``avail_labels`` via the var maps
                cols = []
                if src is not None and self.source_labels is not None:
                    for lab in self.source_labels:
                        cols.append(src[:, sv_map[lab], 0])
                for lab in self.input_labels:
                    cols.append(inp[:, iv_map[lab], 0])
                return np.column_stack(cols)

            # Use the training split when present (that is what models fit).
            if split_info is not None:
                sto_src = train_data["s"] if "s" in train_data else None
                sto_inp = train_data["x"]
            else:
                sto_src, sto_inp = source_np, input_np
            X_stored = _values_matrix(sto_src, sto_inp)
            X_raw = _values_matrix(raw_source_np, raw_input_np)

            # df_adj is [child, parent]; CausalDisco wants W[parent, child].
            W_full = (df_adj.loc[avail_labels, avail_labels].values.T != 0)

            sortability = {
                "definition": (
                    "path-weighted order alignment over directed paths "
                    "(CausalDisco order_alignment_paths; ties count 1/2)"
                ),
                "references": [
                    "Reisach et al. 2021 (arXiv:2102.13647) var-sortability",
                    "Reisach et al. 2023 (arXiv:2303.18211) R2-sortability",
                ],
                "normalize_method": normalize_method if normalize else None,
                "n_samples_stored": int(X_stored.shape[0]),
                "n_samples_raw": int(X_raw.shape[0]),
                "varsortability": var_sortability(X_stored, W_full),
                "varsortability_raw": var_sortability(X_raw, W_full),
                "r2_sortability": r2_sortability(X_stored, W_full),
                "r2_sortability_raw": r2_sortability(X_raw, W_full),
            }
            with open(join(save_dir, "sortability.json"), "w", encoding="utf-8") as file:
                json.dump(sortability, file, indent=2, sort_keys=True, ensure_ascii=False)
            print(
                "Sortability: var={vs:.3f} (raw {vsr:.3f}), "
                "R2={r2s:.3f} (raw {r2r:.3f})".format(
                    vs=sortability["varsortability"],
                    vsr=sortability["varsortability_raw"],
                    r2s=sortability["r2_sortability"],
                    r2r=sortability["r2_sortability_raw"],
                )
            )
        except Exception as exc:  # noqa: BLE001 - analytics must not break generation
            print(f"Warning: sortability computation failed: {exc}")

        # Export dataset metadata for evaluation functions (NEW)
""",
    "sortability-export",
)

P.write_text(t, encoding="utf-8")
print("scm.py patched")
