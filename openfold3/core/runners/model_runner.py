# TODO add license


import pytorch_lightning as pl
import torch

from openfold3.core.utils.exponential_moving_average import ExponentialMovingAverage
from openfold3.core.utils.tensor_utils import tensor_tree_map


class ModelRunnerNotRegisteredError(Exception):
    """A custom error for for unregistered ModelRunners."""

    def __init__(self, model_runner_name: str) -> None:
        super().__init__()
        self.model_runner_name = model_runner_name

    def __str__(self):
        return f"""ModelRunner {self.model_runner_name} missing from model runner \
                registry. Wrap you model runner definition using the \
                model_implementations.registry.register_model decorator."""


# TODO implement shared hooks and methods for OpenFold models
class ModelRunner(pl.LightningModule):
    """High-level LightningModule class implementing hooks shared by OpenFold models.

    For clarity, where possible, follow the hook order specified in the pseudocode
    provided in the PL documentation:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks"""

    def __init__(
        self, model_class: torch.nn.Module, config: dict, _compile: bool = True
    ) -> None:
        """Assign general attributes and initialize the model.

        Args:
            model_class (nn.Module):
                The model class to be used.
            config (dict):
                <Here, need a description of general config structure and
                arguments.>
            _compile (bool):
                Whether to compile the model using torch.compile. Defaults to True.
        """
        super().__init__()
        if not hasattr(self, "_registered"):
            raise ModelRunnerNotRegisteredError(self.__class__.__name__)
        # Save hyperparameters before defining model as recommended here:
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/13615
        self.save_hyperparameters()
        self.config = config

        self.model = (
            torch.compile(model_class(config)) if _compile else model_class(config)
        )

        self.ema = ExponentialMovingAverage(model=self.model, decay=config.ema.decay)
        self.cached_weights = None
        self.last_lr_step = -1

    def forward(self, batch):
        return self.model(batch)

    # TODO refactor training stage logic here
    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}",
                indiv_loss,
                prog_bar=(loss_name == "loss"),
                on_step=train,
                on_epoch=(not train),
                logger=True,
                sync_dist=False,
            )

            if train:
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=False,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, outputs, superimposition_metrics=(not train)
            )

        for k, v in other_metrics.items():
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                prog_bar=(k == "loss"),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
            )

    def training_step(self, batch, batch_idx):
        example_feat = next(
            iter(v for v in batch.values() if isinstance(v, torch.Tensor))
        )
        if self.ema.device != example_feat.device:
            self.ema.to(example_feat.device)

        # Run the model
        outputs = self.model(batch)

        # Compute loss
        loss, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

        # Log it
        self._log(loss_breakdown, batch, outputs)

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def eval_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if self.cached_weights is None:
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling
            # load_state_dict().
            def clone_param(t):
                return t.detach().clone()

            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])

        # Run the model
        outputs = self(batch)

        batch["use_clamped_fape"] = 0.0

        # Compute loss and other metrics
        _, loss_breakdown = self.loss(outputs, batch, _return_breakdown=True)

        self._log(loss_breakdown, batch, outputs, train=False)

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        # TODO implement
        pass

    def configure_optimizers(self):
        pass

    def _compute_validation_metrics(
        self, batch, outputs, superimposition_metrics=False
    ):
        pass
