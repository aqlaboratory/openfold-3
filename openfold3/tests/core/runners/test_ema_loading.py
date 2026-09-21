from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from ml_collections import ConfigDict

from openfold3.core.runners.model_runner import ModelRunner


# ==========
# SETUP
# ==========
class DummyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.register_buffer(
            "version_tensor", torch.tensor([1, 0, 0], dtype=torch.long)
        )

    def forward(self, batch):
        return {"preds": self.linear(batch["x"])}


def get_dummy_config():
    config = ConfigDict()
    config.settings = ConfigDict()
    config.settings.ema = ConfigDict({"decay": 0.999})
    return config


@pytest.fixture
def runner():
    config = get_dummy_config()
    model_runner = ModelRunner(model_class=DummyModel, config=config)
    model_runner.loss = MagicMock(
        return_value=(torch.tensor(1.0), {"loss": torch.tensor(1.0)})
    )
    model_runner._log = MagicMock()
    return model_runner


@pytest.fixture
def dummy_batch():
    return {"x": torch.randn(1, 10)}


# ==========
# TESTS
# ==========
class TestEMALoading:
    def test_ema_no_init_inference(self, runner, dummy_batch):
        """
        Check EMA params remain empty if no training occurs, saving memory
        """
        assert len(runner.ema.params) == 1 and "version_tensor" in runner.ema.params, (
            "EMA params should only contain 'version_tensor' upon initialization."
        )

        runner.eval_step(dummy_batch, batch_idx=0)

        assert len(runner.ema.params) == 1 and "version_tensor" in runner.ema.params, (
            "EMA params should still only contain 'version_tensor' during inference to save memory."
        )

        assert runner.cached_weights is None, (
            "EMA weights should not be loaded, hence no weights should be cached."
        )

        runner.on_validation_epoch_end()
        assert runner.cached_weights is None, "Cached weights were not cleared."

    def test_ema_init_during_training(self, runner, dummy_batch):
        """
        Ensures EMA params are properly cloned when the first training step runs.
        """
        assert len(runner.ema.params) == 1 and "version_tensor" in runner.ema.params, (
            "EMA params should only contain 'version_tensor' upon initialization."
        )

        runner.training_step(dummy_batch, batch_idx=0)

        assert runner.ema.params, "EMA params should be populated after training_step."
        assert (
            len(runner.ema.params) == len(runner.model.state_dict())
            and "linear.weight" in runner.ema.params
        ), "Model parameters were not cloned into EMA."

    def test_ema_weight_swapping_during_validation(self, runner, dummy_batch):
        """
        Ensures the model successfully swaps to EMA weights for validation,
        and restores the live weights afterward.
        """
        runner.training_step(dummy_batch, batch_idx=0)

        original_weight = runner.model.linear.weight.clone()

        # Artificially alter the EMA weights to see if a swap happens
        with torch.no_grad():
            runner.ema.params["linear.weight"].add_(5.0)
        altered_ema_weight = runner.ema.params["linear.weight"].clone()

        runner.eval_step(dummy_batch, batch_idx=0)

        assert torch.allclose(runner.model.linear.weight, altered_ema_weight), (
            "Model failed to load EMA weights during eval_step."
        )
        assert runner.cached_weights is not None, "Original weights were not cached."

        runner.on_validation_epoch_end()

        assert torch.allclose(runner.model.linear.weight, original_weight), (
            "Model failed to restore original weights after validation."
        )
        assert runner.cached_weights is None, "Cached weights were not cleared."
