"""Nodewise MSE labels, epoch aggregation, and calibration logging."""
from types import SimpleNamespace

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from causaliT.training.forecasters.attention_selector_forecaster import AttentionSelectorForecaster
from causaliT.training import dropout_selection as ds


@pytest.mark.parametrize("n_source,labels", [(0, ["X1", "X2", "X3"]),
                                           (1, ["S1", "X1", "X2"])])
def test_node_values_and_labels(n_source, labels):
    logged = {}
    def log(key, value, **kwargs):
        logged[key] = float(value)
        assert kwargs["batch_size"] == 1
        assert kwargs["on_epoch"] and not kwargs["on_step"]
    obj = SimpleNamespace(log=log)
    AttentionSelectorForecaster._log_nodewise_validation_mse(
        obj, torch.tensor([[[1.], [2.], [3.]]]), torch.zeros(1, 3), n_source,
    )
    assert logged == dict(zip([f"val_mse_{x}" for x in labels], [1., 4., 9.]))


def test_epoch_aggregation_weights_short_final_batch():
    class Probe(pl.LightningModule):
        _log_nodewise_validation_mse = AttentionSelectorForecaster._log_nodewise_validation_mse

        def validation_step(self, batch, batch_idx):
            target = batch[0]
            self._log_nodewise_validation_mse(torch.zeros_like(target).unsqueeze(-1), target, 0)

    loader = DataLoader(TensorDataset(torch.tensor([[1.], [1.], [4.]])), batch_size=2)
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    result = trainer.validate(Probe(), dataloaders=loader, verbose=False)
    assert result[0]["val_mse_X1"] == pytest.approx(6.)


def test_warmup_enables_validation_and_candidate_logger(monkeypatch):
    captured = {}
    class Trainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, model, datamodule):
            captured["fit"] = True

    monkeypatch.setattr(ds.pl, "Trainer", Trainer)
    sentinel = object()
    ds._recon_warmup(object(), object(), 500, False, metrics_logger=sentinel)
    assert captured["logger"] is sentinel
    assert captured["max_epochs"] == 500
    assert captured["limit_val_batches"] == 1.0
    assert captured["fit"]