import importlib
import itertools
import logging
import traceback
from pathlib import Path

import torch
from torchmetrics import MeanMetric, MetricCollection, PearsonCorrCoef
from torchmetrics.wrappers import MetricTracker

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

        # TODO: Forcing naming convention to be compatible with older runs
        #  Make consistent later
        # Initialize all training epoch metric objects
        train_metrics = {
            f"{loss_name}_epoch": MeanMetric(nan_strategy="ignore")
            for loss_name in TRAIN_LOSSES
        }
        train_metrics.update(
            {metric_name: MeanMetric(nan_strategy="ignore") for metric_name in METRICS}
        )
        self.train_metrics = MetricCollection(train_metrics, prefix="train/")

        # Initialize all validation epoch metric objects
        val_metrics = {
            metric_name: MeanMetric(nan_strategy="ignore")
            for metric_name in VAL_LOGGED_METRICS
            if metric_name not in CORRELATION_METRICS
        }
        val_metrics.update(
            {
                metric_name: PearsonCorrCoef(num_outputs=1)
                for metric_name in CORRELATION_METRICS
            }
        )
        self.val_metrics = MetricCollection(val_metrics, prefix="val/")

        # Not all metrics will be calculated for each stage of training or between
        # training and validation. Keep track of which metrics are enabled.
        metric_log_names = itertools.chain(
            self.train_metrics.keys(), self.val_metrics.keys()
        )
        self.metric_enabled = {metric_name: False for metric_name in metric_log_names}

        self.tracker = MetricTracker(MeanMetric(nan_strategy="ignore"), maximize=True)

    def _update_epoch_metric(
        self, phase: str, metric_log_name: str, metric_value: torch.Tensor
    ):
        """

        Args:
            phase:
                Phase of training, accepts "train" or "val"
            metric_log_name:
                Name of the metric in the log, including prefix or postfix
            metric_value:
                Value of the metric to update
        """
        if metric_log_name not in self.metric_enabled:
            raise ValueError(
                f"Metric {metric_log_name} is not being tracked and will "
                f"not appear in epoch metrics. Please add it to "
                f"the {phase.upper()}_LOGGED_METRICS constant."
            )

        if not self.metric_enabled[metric_log_name]:
            self.metric_enabled[metric_log_name] = True

        metrics = self.train_metrics if phase == "train" else self.val_metrics
        metric_obj = metrics[metric_log_name]
        metric_obj.update(*metric_value)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"

        for loss_name, indiv_loss in loss_breakdown.items():
            metric_name = f"{phase}/{loss_name}"
            metric_epoch_name = f"{metric_name}_epoch" if train else metric_name

            # Update mean metrics for epoch logging
            self._update_epoch_metric(
                phase=phase,
                metric_log_name=metric_epoch_name,
                metric_value=(indiv_loss,),
            )

            # Only log steps for training
            # Ignore nan losses, where the loss was not applicable for the sample
            if train and not torch.isnan(indiv_loss):
                self.log(
                    metric_name,
                    indiv_loss,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )

        with torch.no_grad():
            if train:
                other_metrics = get_metrics(
                    batch,
                    outputs,
                    superimposition_metrics=False,
                    compute_extra_lddt_metrics=False,
                )
            else:
                # TODO: Model selection - consider replacing this call directly with
                #  model selection call so that we have aggregated statistics of
                #  all the diffusion samples
                other_metrics_per_sample = get_metrics(
                    batch,
                    outputs,
                    superimposition_metrics=True,
                    compute_extra_lddt_metrics=True,
                )

                other_metrics = compute_model_selection_metric(
                    batch=batch,
                    outputs=outputs,
                    metrics=other_metrics_per_sample,
                    weights=self.model_selection_weights,
                )

        for k, v in other_metrics.items():
            mean_metric = torch.mean(v)
            metric_name = f"{phase}/{k}"

            # Update mean metrics for epoch logging
            self._update_epoch_metric(
                phase=phase, metric_log_name=metric_name, metric_value=(mean_metric,)
            )

            # TODO: Maybe remove this extra logging
            # Only log steps for training
            # Ignore nan metric, where the metric was not applicable for the sample
            if train and not torch.isnan(mean_metric):
                self.log(
                    f"{metric_name}_step",
                    mean_metric,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )
        if not train:
            for molecule_type in ["protein", "rna", "dna", "ligand", "complex"]:
                plddt_key = f"plddt_{molecule_type}"
                lddt_key = f"lddt_intra_{molecule_type}"
                if plddt_key in other_metrics and lddt_key in other_metrics:
                    plddt = other_metrics[plddt_key].flatten()
                    lddt = other_metrics[lddt_key].flatten()
                    metric_name = f"val/pearson_correlation_lddt_plddt_{molecule_type}"
                    self._update_epoch_metric(
                        phase=phase,
                        metric_log_name=metric_name,
                        metric_value=(lddt, plddt),
                    )

            self.tracker.update(other_metrics["model_selection"])

    def training_step(self, batch, batch_idx):
        example_feat = next(
            iter(v for v in batch.values() if isinstance(v, torch.Tensor))
        )
        if self.ema.device != example_feat.device:
            self.ema.to(example_feat.device)

        # TODO: Remove debug logic
        pdb_id = ", ".join(batch.pop("pdb_id"))
        logger.warning(
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

        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(
                "-" * 40
                + "\n"
                + f"Train step failed with pdb id {pdb_id}: {str(e)}\n"
                + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                + "-" * 40
            )
            raise e

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
        logger.warning(
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

        except Exception as e:
            tb = traceback.format_exc()  # Get the full traceback
            logger.warning(
                "-" * 40
                + "\n"
                + f"Train step failed with pdb id {', '.join(pdb_id)}: {str(e)}\n"
                + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                + "-" * 40
            )
            raise e

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

    def on_train_epoch_start(self):
        # At the start of each virtual epoch we want to resample the set of
        # datapoints to train on
        self.trainer.train_dataloader.dataset.resample_epoch()

    def on_validation_epoch_start(self):
        self.tracker.increment()

    def _log_epoch_metrics(self, metrics: MetricCollection):
        """Log aggregated epoch metrics for training or validation.

        Args:
            metrics: MetricCollection object containing the metrics to log
        """
        for metric_name, metric_obj in metrics.items():
            # Only log metrics that have been updated
            if self.metric_enabled.get(metric_name):
                if not self.trainer.sanity_checking:
                    # Sync and reduce metric across ranks
                    output = metric_obj.compute()
                    self.log(
                        metric_name,
                        output,
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        sync_dist=False,  # Already synced in compute()
                    )

                # Reset metric for next epoch
                metric_obj.reset()

    def _log_epoch_tracker(self):
        """Log the tracker metric for the epoch."""
        if not self.trainer.sanity_checking:
            best_score, which_epoch = self.tracker.best_metric(return_step=True)
            self.log(
                "val/model_selection_metric",
                self.tracker.compute(),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
            )
            self.log(
                "val/best_model_selection_metric",
                best_score,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
            )
            self.log(
                "val/best_model_selection_epoch",
                which_epoch,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
            )

    def on_train_epoch_end(self):
        """Log aggregated epoch metrics for training."""
        self._log_epoch_metrics(metrics=self.train_metrics)

    def on_validation_epoch_end(self):
        """Log aggregated epoch metrics for validation."""
        self._log_epoch_metrics(metrics=self.val_metrics)
        self._log_epoch_tracker()

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
