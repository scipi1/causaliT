"""Quick check that gradient clipping is wired up correctly."""
import inspect
from causaliT.training.forecasters.variance_causal_forecaster import VarianceCausalForecaster
from causaliT.training.trainer import train_single_fold

src = inspect.getsource(VarianceCausalForecaster.training_step)
src2 = inspect.getsource(train_single_fold)

assert "clip_grad_norm_" in src, "MISSING: clip_grad_norm_ not in forecaster.training_step"
assert "gradient_clip_val" in src, "MISSING: gradient_clip_val not in forecaster.training_step"
assert "gradient_clip_val" in src2, "MISSING: gradient_clip_val not in pl.Trainer call"

print("✓ forecaster.training_step: manual clip_grad_norm_ present")
print("✓ trainer.train_single_fold: gradient_clip_val passed to pl.Trainer")
print("All gradient clipping checks passed.")
