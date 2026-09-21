from pathlib import Path
p = Path("causaliT/training/forecasters/attention_selector_forecaster.py")
src = p.read_text(encoding="utf-8")

# In on_load_checkpoint: drop oracle_shd_gt from the checkpoint when the
# current model did not register the buffer (eval loading without data_dir).
old = '''        # Restore the HSIC-constraint dual state (absent in checkpoints that
        # predate the feature -> keep the freshly-initialised values).'''
new = '''        # Eval-loading path: the model skipped registering ``oracle_shd_gt``
        # (data_dir=None); drop the checkpoint\'s copy so strict loading works.
        if "oracle_shd_gt" not in current_keys:
            checkpoint["state_dict"].pop("oracle_shd_gt", None)

        # Restore the HSIC-constraint dual state (absent in checkpoints that
        # predate the feature -> keep the freshly-initialised values).'''
assert src.count(old) == 1, "anchor"
src = src.replace(old, new)
p.write_text(src, encoding="utf-8")
print("ok")
