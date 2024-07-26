# TODO add license


import pytorch_lightning as pl
import torch

from openfold3.core.loss.loss import lddt_ca
from openfold3.core.np import residue_constants
from openfold3.core.utils.exponential_moving_average import ExponentialMovingAverage
from openfold3.core.utils.superimposition import superimpose
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.core.utils.validation_metrics import (
    drmsd,
    gdt_ha,
    gdt_ts,
)


# TODO implement shared hooks and methods for OpenFold models
class ModelRunner(pl.LightningModule):
    """High-level LightningModule class implementing hooks shared by OpenFold models.

    For clarity, where possible, follow the hook order specified in the pseudocode
    provided in the PL documentation:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks"""

    # QUESTION do we want to enforce class registration with this decorator? Part B
    # def __init__(self) -> None:
    #     if not self.__class__._registered:
    #         raise DatasetNotRegisteredError()

    def __init__(self, model_class: torch.nn.Module, config: dict) -> None:
        """Assign general attributes and initialize the model.

        Args:
            config (dict):
                <Here, need a description of general config structure and
                arguments.>
        """
        super().__init__()
        # Save hyperparameters before defining model as recommended here:
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/13615
        self.save_hyperparameters()
        self.config = config
        self.model = model_class(config)
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
        if self.ema.device != batch["aatype"].device:
            self.ema.to(batch["aatype"].device)

        # Run the model
        outputs = self.model(batch)

        # Compute loss
        loss, loss_breakdown = self.loss(outputs, batch, _return_breakdown=True)

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

    def on_train_epoch_start(self) -> None:
        """Resample epoch_len number of samples for the training datasets at the start
        of each epoch."""
        self.trainer.train_dataloader.dataset.resample_epoch()

    # def transfer_batch_to_device(
    #     self, batch: Any, device: device, dataloader_idx: int
    # ) -> Any:
    #     """Device-transfer logic for mid-forward-pass recycling with CPU offloading.

    #     This overwrites the default transfer_batch_to_device hook, which transfers
    #     the entire batch to the device. This hook is necessary to transfer only the
    #     constant features to the device and keep the recycling features on the CPU.
    #     The batch parser method of the wrapper needs to call the recycling features
    #     from the self._recycling_features attribute and the nn.Module's forward method
    #     needs to specify the device transfers logic for sub-forward-pass recycling.

    #     Args:
    #         batch (Any):
    #             batch containing constant and recycled feature tensors
    #         device (device):
    #             _description_
    #         dataloader_idx (int):
    #             _description_

    #     Returns:
    #         dict: constant feature tensors + first set of recycling features
    #     """
    #     self._recycling_features = batch[
    #         "recycling_features"
    #     ]  # TODO change key if FeatureDict structure changes
    #     # This workaround is fine for multi-GPU runs with DDP and across
    #     # multiple samples

    #     # Only return constant features on the batch
    #     batch = {
    #         "constant_features": {
    #             k: v.to(device) for k, v in batch["constant_features"].items()
    #         }
    #     }
    #     return batch
    #     # This is necessary because keeping the recycling features in the same feature
    #     # dict as the constant features results in their transfer to GPU by the time
    #     # the batch variable gets to LightningModule.training_step
    #     # This is likely due to a bug in the training step call stack ...
