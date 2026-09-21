"""Patch part 2: GT loading call + _load_oracle_shd_gt method + _step branch."""
from pathlib import Path

p = Path("causaliT/training/forecasters/attention_selector_forecaster.py")
src = p.read_text(encoding="utf-8")

# P2: load GT after the (skipped) hard-mask loading block.
old = '''            )

        self.save_hyperparameters(config)'''
new = '''            )

        # Oracle-SHD constraint: GT adjacency for the constraint monitor,
        # loaded into a dedicated buffer that NEVER reaches the forward pass.
        if self.hsic_constraint_source == "oracle_shd":
            if data_dir is None:
                raise ValueError(
                    "hsic_constraint.source=\'oracle_shd\' requires data_dir "
                    "(pass via create_model_instance) to load the GT DAG."
                )
            self._load_oracle_shd_gt(config, data_dir)

        self.save_hyperparameters(config)'''
assert src.count(old) == 1, "P2 anchor"
src = src.replace(old, new)

# P3: loader method right after _load_combined_oracle_mask.
old = '''f"{\', homogeneous square layout\' if self.homogeneous_nodes else \'\'})"
        )
'''
new = old + '''
    def _load_oracle_shd_gt(self, config: dict, data_dir: str):
        """Load the GT DAG adjacency for the oracle-SHD constraint.

        Registered as the dedicated ``oracle_shd_gt`` buffer, which is NEVER
        passed to ``forward`` (the __init__ validation hard-errors when
        use_hard_masks / use_oracle_attention are on, so no GT can leak into
        the attention).  Same layout convention as ``oracle_combined_mask``:
        square (N, N) in homogeneous mode, (L_X, L_S+L_X) in split mode;
        entry [i, j] = 1 iff j is a parent of i.
        """
        mask_files = config["training"].get("hard_mask_files", None)
        if mask_files is None:
            raise ValueError(
                "hsic_constraint.source=\'oracle_shd\' requires "
                "training.hard_mask_files (dec_cross / dec_self GT CSVs)."
            )
        dataset_dir = join(data_dir, config["data"]["dataset"])
        masks = load_dag_masks(dataset_dir, mask_files, device="cpu")
        if masks is None:
            raise ValueError(
                f"oracle_shd: no DAG mask files found in {dataset_dir}."
            )
        cross_mask = masks.get("dec_cross", None)
        self_mask = masks.get("dec_self", None)
        if cross_mask is None or self_mask is None:
            raise ValueError(
                "oracle_shd: expected \'dec_cross\' and \'dec_self\' masks."
            )
        if self.homogeneous_nodes:
            gt = torch.zeros(self.N, self.N, dtype=cross_mask.dtype)
            gt[self.S_seq_len :, : self.S_seq_len] = cross_mask
            gt[self.S_seq_len :, self.S_seq_len :] = self_mask
        else:
            gt = torch.cat([cross_mask, self_mask], dim=1)
        self.register_buffer("oracle_shd_gt", gt)
        print(
            f"[oracle_shd] GT adjacency loaded: shape {tuple(gt.shape)} "
            f"({int(gt.sum())} edges)"
        )
'''
assert src.count(old) == 1, "P3 anchor"
src = src.replace(old, new)

# P4a: constraint-value source switch in _step.
old = '''        if self.hsic_constraint_enabled:
            # Lagrangian / augmented-Lagrangian constraint term:'''
new = '''        # Constraint source switch: the monitored quantity is the HSIC
        # (default) or, with source=\'oracle_shd\', the EXPECTED SHD to the GT
        # DAG under the directed gate posterior (mean per-pair
        # misclassification; differentiable through the same Q/K structural
        # pathway as L0).  HSIC is still computed and logged either way --
        # under oracle_shd it is a pure EVALUATION metric.
        constraint_value = hsic_value
        if (
            self.hsic_constraint_enabled
            and self.hsic_constraint_source == "oracle_shd"
        ):
            post = attention_weights.mean(dim=0)
            gt = self.oracle_shd_gt.to(dtype=post.dtype)
            if post.shape != gt.shape:
                raise ValueError(
                    f"oracle_shd: posterior {tuple(post.shape)} != GT "
                    f"{tuple(gt.shape)}"
                )
            constraint_value = (gt * (1.0 - post) + (1.0 - gt) * post).mean()
            self.log(f"{stage}_oracle_shd", constraint_value,
                     on_step=False, on_epoch=True)
        if self.hsic_constraint_enabled:
            # Lagrangian / augmented-Lagrangian constraint term:'''
assert src.count(old) == 1, "P4a anchor"
src = src.replace(old, new)

# P4b: the violation and the EMA track the constraint value, not HSIC.
old = "            hsic_violation = hsic_value - self.hsic_tol"
assert src.count(old) == 1, "P4b anchor"
src = src.replace(old, "            hsic_violation = constraint_value - self.hsic_tol")
old = "                v = float(hsic_value.detach())"
assert src.count(old) == 1, "P4c anchor"
src = src.replace(old, "                v = float(constraint_value.detach())")

p.write_text(src, encoding="utf-8")
print("P2-P4 ok")
