import importlib
import itertools
import logging
from pathlib import Path

import torch
from torchmetrics import MeanMetric, MetricCollection, PearsonCorrCoef

from openfold3.core.loss.loss_module import AlphaFold3Loss
from openfold3.core.metrics.confidence import (
    compute_plddt,
    compute_predicted_aligned_error,
    compute_predicted_distance_error,
    compute_weighted_ptm,
)
from openfold3.core.metrics.model_selection import compute_model_selection_metric
from openfold3.core.metrics.validation_all_atom import (
    get_metrics,
)
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.atomize_utils import get_token_frame_atoms
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.af3_all_atom.config.base_config import (
    model_selection_metric_weights_config,
    project_config,
)
from openfold3.projects.af3_all_atom.config.dataset_config_builder import (
    AF3DatasetConfigBuilder,
)
from openfold3.projects.af3_all_atom.constants import (
    CORRELATION_METRICS,
    METRICS,
    TRAIN_LOSSES,
    VAL_LOGGED_METRICS,
    VAL_LOSSES,
)
from openfold3.projects.af3_all_atom.model import AlphaFold3
from openfold3.projects.registry import register_project

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    from deepspeed.ops.adam import DeepSpeedCPUAdam

logger = logging.getLogger(__name__)

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


@register_project(
    "af3_all_atom", AF3DatasetConfigBuilder, project_config, REFERENCE_CONFIG_PATH
)
class AlphaFold3AllAtom(ModelRunner):
    def __init__(self, model_config, _compile=True):
        super().__init__(model_class=AlphaFold3, config=model_config, _compile=_compile)

        self.loss = (
            torch.compile(AlphaFold3Loss(config=model_config.architecture.loss_module))
            if _compile
            else AlphaFold3Loss(config=model_config.architecture.loss_module)
        )

        self.model_selection_weights = model_selection_metric_weights_config[
            self.config.settings.model_selection_weight_scheme
        ]

        self._setup_train_metrics()
        self._setup_val_metrics()
        self._init_metric_enabled_tracker()

    def _setup_train_metrics(self):
        """Set up training loss and metric collection objects."""

        # TODO: Forcing naming convention to be compatible with older runs
        #  Make consistent later
        # Initialize all training epoch metric objects
        train_losses = {
            loss_name: MeanMetric(nan_strategy="ignore") for loss_name in TRAIN_LOSSES
        }
        self.train_losses = MetricCollection(
            train_losses, prefix="train/", postfix="_epoch"
        )

        train_metrics = {
            metric_name: MeanMetric(nan_strategy="ignore") for metric_name in METRICS
        }

        self.train_metrics = MetricCollection(train_metrics, prefix="train/")

    def _setup_val_metrics(self):
        """Set up validation loss and metric collection objects."""

        # Initialize all validation epoch metric objects
        val_losses = {
            loss_name: MeanMetric(nan_strategy="ignore") for loss_name in VAL_LOSSES
        }
        self.val_losses = MetricCollection(val_losses, prefix="val/")

        val_metrics = {
            metric_name: MeanMetric(nan_strategy="ignore")
            for metric_name in VAL_LOGGED_METRICS
        }
        val_metrics.update(
            {
                metric_name: PearsonCorrCoef(num_outputs=1)
                for metric_name in CORRELATION_METRICS
            }
        )
        self.val_metrics = MetricCollection(val_metrics, prefix="val/")

    def _init_metric_enabled_tracker(self):
        """
        Initialize map of enabled losses and metrics for logging. Losses default to
        False because not all losses will be calculated for each stage of training.
        The appropriate losses will be enabled after the first pass through the model.
        """
        loss_log_names = itertools.chain(
            self.train_losses.keys(), self.val_losses.keys()
        )
        metric_log_names = itertools.chain(
            self.train_metrics.keys(), self.val_metrics.keys()
        )
        metric_enabled = {loss_name: False for loss_name in loss_log_names}
        metric_enabled.update({metric_name: True for metric_name in metric_log_names})
        self.metric_enabled = metric_enabled

    def _update_epoch_metric(
        self,
        phase: str,
        metric_log_name: str,
        metric_value: [torch.Tensor, tuple],
        metric_collection: MetricCollection,
    ):
        """Update metrics for the epoch logging.

        Args:
            phase:
                Phase of training, accepts "train" or "val"
            metric_log_name:
                Name of the metric in the log, including prefix or postfix
            metric_value:
                Value of the metric to update
            metric_collection:
                MetricCollection object containing the metric to update
        """
        if metric_log_name not in self.metric_enabled:
            raise ValueError(
                f"Metric {metric_log_name} is not being tracked and will "
                f"not appear in epoch metrics. Please add it to "
                f"the {phase.upper()}_LOSSES or METRICS constants."
            )

        if not self.metric_enabled[metric_log_name]:
            self.metric_enabled[metric_log_name] = True

        metric_obj = metric_collection[metric_log_name]

        metric_value = (
            (metric_value,) if type(metric_value) is not tuple else metric_value
        )
        metric_obj.update(*metric_value)

    def _get_metrics(self, batch, outputs, train=True) -> dict:
        with torch.no_grad():
            if train:
                return get_metrics(
                    batch,
                    outputs,
                    compute_extra_val_metrics=False,
                )

            # TODO: Model selection - consider replacing this call directly with
            #  model selection call so that we have aggregated statistics of
            #  all the diffusion samples
            metrics_per_sample = get_metrics(
                batch,
                outputs,
                compute_extra_val_metrics=True,
            )

            metrics = compute_model_selection_metric(
                outputs=outputs,
                metrics=metrics_per_sample,
                weights=self.model_selection_weights,
                pdb_id=batch["pdb_id"],
            )

            for metric_name in CORRELATION_METRICS:
                molecule_type = metric_name.split("_")[-1]
                plddt_key = f"plddt_{molecule_type}"
                lddt_key = f"lddt_intra_{molecule_type}"

                plddt = metrics.get(plddt_key)
                lddt = metrics.get(lddt_key)

                if plddt is not None and lddt is not None:
                    metrics[metric_name] = (lddt, plddt)

            logger.debug(
                f"Validation sample {', '.join(batch['pdb_id'])} on rank "
                f"{self.global_rank} has the following metrics: "
                f"{', '.join(list(metrics.keys()))}"
            )

            return metrics

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"

        metrics = self._get_metrics(batch, outputs, train=train)

        loss_collection = self.train_losses if phase == "train" else self.val_losses
        for loss_name, indiv_loss in loss_breakdown.items():
            metric_log_name = f"{phase}/{loss_name}"
            metric_epoch_name = f"{metric_log_name}_epoch" if train else metric_log_name

            # Mean over sample and batch dims
            indiv_loss = indiv_loss.mean()

            # Update mean metrics for epoch logging
            self._update_epoch_metric(
                phase=phase,
                metric_log_name=metric_epoch_name,
                metric_value=indiv_loss,
                metric_collection=loss_collection,
            )

            # Only log steps for training
            # Ignore nan losses, where the loss was not applicable for the sample
            if train and not torch.isnan(indiv_loss):
                self.log(
                    metric_log_name,
                    indiv_loss,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )

        metric_collection = self.train_metrics if phase == "train" else self.val_metrics
        for metric_name, metric_value in metrics.items():
            metric_log_name = f"{phase}/{metric_name}"

            # Update mean metrics for epoch logging
            self._update_epoch_metric(
                phase=phase,
                metric_log_name=metric_log_name,
                metric_value=metric_value,
                metric_collection=metric_collection,
            )

            # TODO: Maybe remove this extra logging
            # Only log steps for training
            # Ignore nan metric, where the metric was not applicable for the sample
            if train and not torch.isnan(metric_value).any():
                self.log(
                    f"{metric_log_name}_step",
                    metric_value,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )

    def training_step(self, batch, batch_idx):
        example_feat = next(
            iter(v for v in batch.values() if isinstance(v, torch.Tensor))
        )
        if self.ema.device != example_feat.device:
            self.ema.to(example_feat.device)

        # TODO: Remove debug logic
        pdb_id = ", ".join(batch.pop("pdb_id"))
        logger.debug(
            f"Started model forward pass for {pdb_id} on rank {self.global_rank} "
            f"step {self.global_step}"
        )

        try:
            # Run the model
            batch, outputs = self.model(batch)

            # Compute loss
            loss, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

            # Log it
            self._log(loss_breakdown, batch, outputs)

        except Exception:
            logger.exception(f"Train step failed with pdb id {pdb_id}")
            raise

        return loss

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

        # TODO: Remove debug logic
        pdb_id = batch.pop("pdb_id")
        atom_array = batch.pop("atom_array")
        logger.debug(
            f"Started validation for {', '.join(pdb_id)} on rank {self.global_rank} "
            f"step {self.global_step}"
        )

        try:
            # Run the model
            batch, outputs = self(batch)

            # Compute loss and other metrics
            _, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

            batch["atom_array"] = atom_array
            batch["pdb_id"] = pdb_id

            self._log(loss_breakdown, batch, outputs, train=False)

        except Exception:
            logger.exception(f"Validation step failed with pdb id {pdb_id}")
            raise

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # TODO: Remove debug logic
        pdb_id = batch.pop("pdb_id")
        atom_array = batch.pop("atom_array") if "atom_array" in batch else None

        # This is to avoid slow loading for nested dicts in PL
        # Less frequent hanging when non_blocking=True on H200
        # TODO: Determine if this is really needed given other
        #  recent hanging fixes
        def to_device(t):
            return t.to(device=device, non_blocking=True)

        batch = tensor_tree_map(to_device, batch)
        batch["pdb_id"] = pdb_id

        # Add atom array back to the batch if we removed it earlier
        if atom_array:
            batch["atom_array"] = atom_array

        return batch

    def _save_train_dataset_state_to_datamodule(self):
        self.trainer.datamodule.next_dataset_indices = (
            self.trainer.train_dataloader.dataset.next_dataset_indices
        )

    def _load_train_dataset_state_from_datamodule(self):
        self.trainer.train_dataloader.dataset.next_dataset_indices = (
            self.trainer.datamodule.next_dataset_indices
        )

    def on_train_start(self):
        # Reload state from datamodule in case checkpoint has been used
        self._load_train_dataset_state_from_datamodule()
        if self.global_rank == 0:
            logger.debug(
                f"Train start, setting up "
                f"{self.trainer.train_dataloader.dataset.next_dataset_indices=}"
            )

    def on_train_epoch_start(self):
        # At the start of each virtual epoch we want to resample the set of
        # datapoints to train on
        self.trainer.train_dataloader.dataset.resample_epoch()
        self._save_train_dataset_state_to_datamodule()
        if self.global_rank == 0:
            logger.debug(
                "Sampled batch indices: "
                f"{self.trainer.train_dataloader.dataset.indices=}"
            )

    def _log_epoch_metrics(self, metrics: MetricCollection):
        """Log aggregated epoch metrics for training or validation.

        Args:
            metrics: MetricCollection object containing the metrics to log
        """
        # Sync and reduce metrics across ranks
        metrics_output = metrics.compute()
        for name, result in metrics_output.items():
            # Only log metrics that have been updated
            if self.metric_enabled.get(name):
                self.log(
                    name,
                    result,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=False,  # Already synced in compute()
                )

        # Reset metrics for next epoch
        metrics.reset()

    def on_train_epoch_end(self):
        """Log aggregated epoch metrics for training."""
        self._log_epoch_metrics(metrics=self.train_losses)
        self._log_epoch_metrics(metrics=self.train_metrics)

    def on_validation_epoch_end(self):
        """Log aggregated epoch metrics for validation."""
        if not self.trainer.sanity_checking:
            self._log_epoch_metrics(metrics=self.val_losses)
            self._log_epoch_metrics(metrics=self.val_metrics)

    def configure_optimizers(
        self,
        learning_rate: float = 1.8e-3,
    ) -> dict:
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer_config = self.config.settings.optimizer

        if deepspeed_is_installed and optimizer_config.use_deepspeed_adam:
            optimizer = DeepSpeedCPUAdam(
                self.parameters(),
                lr=learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
                adamw_mode=False,
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
            )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = learning_rate

        lr_scheduler = AlphaFoldLRScheduler(
            optimizer, last_epoch=self.last_lr_step, max_lr=learning_rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            },
        }

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        self.ema.load_state_dict(ema)

    # TODO: Integrate with prediction step
    def _compute_confidence_scores(self, batch: dict, outputs: dict) -> dict:
        """Compute confidence metrics. This function is called during inference.

        Args:
            batch (dict):
                Input feature dictionary
            outputs (dict:
                Output dictionary containing the predicted trunk embeddings,
                all-atom positions, and distogram head logits

        Returns:
            confidence_scores (dict):
                Dict containing the following confidence measures:
                pLDDT, PDE, PAE, pTM, iPTM, weighted pTM
        """
        # Used in modified residue ranking
        confidence_scores = {}
        confidence_scores["plddt"] = compute_plddt(outputs["plddt_logits"])
        confidence_scores.update(
            compute_predicted_distance_error(
                outputs["pde_logits"],
                **self.config.confidence.pde,
            )
        )

        if self.config.architecture.heads.pae.enabled:
            confidence_scores.update(
                compute_predicted_aligned_error(
                    outputs["pae_logits"],
                    **self.config.confidence.pae,
                )
            )

            _, valid_frame_mask = get_token_frame_atoms(
                batch=batch,
                x=outputs["atom_positions_predicted"],
                atom_mask=batch["atom_mask"],
            )

            # Compute weighted pTM score
            # Uses pae_logits (SI pg. 27)
            ptm_scores = compute_weighted_ptm(
                logits=outputs["pae_logits"],
                asym_id=batch["asym_id"],
                mask=valid_frame_mask,
                **self.config.confidence.ptm,
            )
            confidence_scores.update(ptm_scores)

        return confidence_scores
