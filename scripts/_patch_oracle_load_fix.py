from pathlib import Path
p = Path("causaliT/training/forecasters/attention_selector_forecaster.py")
src = p.read_text(encoding="utf-8")

# Fix: oracle_shd must not raise on load_from_checkpoint without data_dir
# (eval notebooks load checkpoints with data_dir=None; the GT buffer is only
# needed to TRAIN with the constraint).
old = '''        if self.hsic_constraint_source == "oracle_shd":
            if data_dir is None:
                raise ValueError(
                    "hsic_constraint.source=\'oracle_shd\' requires data_dir "
                    "(pass via create_model_instance) to load the GT DAG."
                )
            self._load_oracle_shd_gt(config, data_dir)'''
new = '''        if self.hsic_constraint_source == "oracle_shd":
            if data_dir is None:
                # Eval/notebook loading path: the GT buffer is only needed to
                # TRAIN with the constraint; skip with a warning instead of
                # raising so load_from_checkpoint works without data_dir.
                logger.warning(
                    "hsic_constraint.source=\'oracle_shd\' but data_dir is "
                    "None: GT buffer not loaded (fine for evaluation; "
                    "training with the constraint would fail)."
                )
            else:
                self._load_oracle_shd_gt(config, data_dir)'''
assert src.count(old) == 1, "P-fix anchor"
src = src.replace(old, new)

# Guard in _step: informative error if the buffer is missing.
old = '''            post = attention_weights.mean(dim=0)
            gt = self.oracle_shd_gt.to(dtype=post.dtype)'''
new = '''            post = attention_weights.mean(dim=0)
            gt_buf = getattr(self, "oracle_shd_gt", None)
            if gt_buf is None:
                raise ValueError(
                    "hsic_constraint.source=\'oracle_shd\': GT buffer not "
                    "loaded (data_dir was None at init)."
                )
            gt = gt_buf.to(dtype=post.dtype)'''
assert src.count(old) == 1, "guard anchor"
src = src.replace(old, new)

p.write_text(src, encoding="utf-8")
print("fix applied")
