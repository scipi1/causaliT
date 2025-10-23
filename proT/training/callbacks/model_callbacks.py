# Third-party imports
import torch
from pytorch_lightning import Callback, Trainer, LightningModule


class GradientLogger(Callback):
    """
    Logs ‖∇θ‖₂ and variance layer‑by‑layer.
    Optionally stores raw gradients as .pt files.
    """
    def __init__(self):
        super().__init__()        

    @staticmethod
    def _stats(t: torch.Tensor):
        return dict(
            grad_norm = t.norm().item(),
        )

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        self.metrics = {}
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            s = self._stats(p.grad.detach())
            self.metrics[f"grad_norm/{name}"] = s["grad_norm"]
    
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log_dict(
                self.metrics,
                on_step=False,
                on_epoch=True
            )
        

class MetricsAggregator(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.logger.log_metrics(
            trainer.callback_metrics,
            step=trainer.current_epoch
        )
