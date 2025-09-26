import gc
import importlib
import itertools
import logging
import traceback
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from torchmetrics import MeanMetric, MetricCollection, PearsonCorrCoef

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.metrics.confidence import (
    compute_global_predicted_distance_error,
    compute_plddt,
    compute_predicted_aligned_error,
    compute_predicted_distance_error,
    compute_weighted_ptm,
)
from openfold3.core.metrics.model_selection import (
    compute_final_model_selection_metric,
    compute_valid_model_selection_metrics,
)
from openfold3.core.metrics.validation_all_atom import (
    get_metrics,
    get_metrics_chunked,
)
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.atomize_utils import get_token_frame_atoms
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.config.model_config import (
    model_selection_metric_weights_config,
)
from openfold3.projects.of3_all_atom.constants import (
    CORRELATION_METRICS,
    TRAIN_LOGGED_METRICS,
    TRAIN_LOSSES,
    VAL_LOGGED_METRICS,
    VAL_LOSSES,
)
from openfold3.projects.of3_all_atom.model import OpenFold3

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    from deepspeed.ops.adam import DeepSpeedCPUAdam

logger = logging.getLogger(__name__)

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


class OpenFold3AllAtom(ModelRunner):
    def __init__(self, model_config, output_dir: Path = None):
        super().__init__(model_class=OpenFold3, config=model_config)

        self.output_dir = output_dir

        self.loss = OpenFold3Loss(config=model_config.architecture.loss_module)

        self.model_selection_weights = model_selection_metric_weights_config[
            self.config.settings.model_selection_weight_scheme
        ]

        self._setup_train_metrics()
        self._setup_val_metrics()
        self._init_metric_enabled_tracker()

    def reseed(self, seed):
        pl.seed_everything(seed)

    def _setup_train_metrics(self):
        """Set up training loss and metric collection objects."""

        # TODO: Forcing naming convention to be compatible with older runs
        #  Make consistent later
        # Initialize all training epoch metric objects
        train_losses = {
            loss_name: MeanMetric(nan_strategy="warn") for loss_name in TRAIN_LOSSES
        }
        self.train_losses = MetricCollection(
            train_losses, prefix="train/", postfix="_epoch"
        )

        train_metrics = {
            metric_name: MeanMetric(nan_strategy="warn")
            for metric_name in TRAIN_LOGGED_METRICS
        }

        self.train_metrics = MetricCollection(train_metrics, prefix="train/")

    def _setup_val_metrics(self):
        """Set up validation loss and metric collection objects."""

        # Initialize all validation epoch metric objects
        val_losses = {
            loss_name: MeanMetric(nan_strategy="warn") for loss_name in VAL_LOSSES
        }
        self.val_losses = MetricCollection(val_losses, prefix="val/")

        val_metrics = {
            metric_name: MeanMetric(nan_strategy="warn")
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
                    compute_lig_diffusion_metrics=True,
                    compute_extra_val_metrics=False,
                )

            num_samples = (
                self.config.architecture.shared.diffusion.no_full_rollout_samples
            )
            num_atoms = outputs["atom_positions_predicted"].shape[-2]
            chunk_metrics_computation = (
                num_samples > 1
                and self.config.settings.memory.eval.per_sample_atom_cutoff is not None
                and num_atoms > self.config.settings.memory.eval.per_sample_atom_cutoff
            )

            if chunk_metrics_computation:
                metrics_per_sample = get_metrics_chunked(
                    batch,
                    outputs,
                    compute_extra_val_metrics=True,
                )
            else:
                metrics_per_sample = get_metrics(
                    batch,
                    outputs,
                    compute_extra_val_metrics=True,
                )

            metrics = compute_valid_model_selection_metrics(
                confidence_config=self.config.confidence,
                outputs=outputs,
                metrics=metrics_per_sample,
            )

            for metric_name in CORRELATION_METRICS:
                molecule_type = metric_name.split("_")[-1]
                plddt_key = f"plddt_{molecule_type}"
                lddt_key = f"lddt_intra_{molecule_type}"

                plddt = metrics_per_sample.get(plddt_key)
                lddt = metrics_per_sample.get(lddt_key)

                if plddt is not None and lddt is not None:
                    plddt = plddt.reshape((-1, 1))
                    lddt = lddt.reshape((-1, 1))
                    metrics[metric_name] = (lddt, plddt)

            return metrics

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"

        metrics = self._get_metrics(batch, outputs, train=train)

        loss_collection = self.train_losses if phase == "train" else self.val_losses
        for loss_name, indiv_loss in loss_breakdown.items():
            metric_log_name = f"{phase}/{loss_name}"
            metric_epoch_name = f"{metric_log_name}_epoch" if train else metric_log_name

            # Update mean metrics for epoch logging
            self._update_epoch_metric(
                phase=phase,
                metric_log_name=metric_epoch_name,
                metric_value=indiv_loss,
                metric_collection=loss_collection,
            )

            # Only log steps for training
            if train:
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
            if train:
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

        pdb_id = ", ".join(batch["pdb_id"])
        preferred_chain_or_interface = batch["preferred_chain_or_interface"]
        logger.debug(
            f"Started model forward pass for {pdb_id} with preferred chain or "
            f"interface {preferred_chain_or_interface} on rank {self.global_rank} "
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
            logger.exception(
                f"Train step failed with pdb id {pdb_id} with "
                f"preferred chain or interface {preferred_chain_or_interface}"
            )
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

        pdb_id = batch["pdb_id"]
        is_repeated_sample = batch.get("repeated_sample").item()
        logger.debug(
            f"Started validation for {', '.join(pdb_id)} on rank {self.global_rank} "
            f"step {self.global_step}, repeated: {is_repeated_sample}"
        )

        try:
            # Run the model
            batch, outputs = self(batch)

            # Compute loss and other metrics
            _, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

            if not is_repeated_sample:
                self._log(loss_breakdown, batch, outputs, train=False)

        except Exception:
            logger.exception(f"Validation step failed with pdb id {', '.join(pdb_id)}")
            raise

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

    def _log_epoch_metrics(
        self, metrics: MetricCollection, compute_model_selection: bool = False
    ):
        """Log aggregated epoch metrics for training or validation.

        Args:
            metrics: MetricCollection object containing the metrics to log
        """
        if not self.trainer.sanity_checking:
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

            if compute_model_selection:
                model_selection = compute_final_model_selection_metric(
                    metrics=metrics_output,
                    model_selection_weights=self.model_selection_weights,
                )

                self.log(
                    "val/model_selection",
                    model_selection,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=False,
                )

        # Reset metrics for next epoch
        metrics.reset()

    def on_train_epoch_end(self):
        """Log aggregated epoch metrics for training."""
        self._log_epoch_metrics(metrics=self.train_losses)
        self._log_epoch_metrics(metrics=self.train_metrics)

    def on_validation_epoch_end(self):
        """Log aggregated epoch metrics for validation."""
        self._log_epoch_metrics(metrics=self.val_losses)
        self._log_epoch_metrics(metrics=self.val_metrics, compute_model_selection=True)

        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

        # Temp fix for val dataloader worker seg fault issues
        # TODO: Figure out why this is not being cleaned up properly
        gc.collect()
        torch.cuda.empty_cache()
        self.trainer.strategy.barrier()

    def configure_optimizers(self) -> dict:
        optimizer_config = self.config.settings.optimizer

        if deepspeed_is_installed and optimizer_config.use_deepspeed_adam:
            optimizer = DeepSpeedCPUAdam(
                self.parameters(),
                lr=optimizer_config.learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
                adamw_mode=False,
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.learning_rate,
                betas=(optimizer_config.beta1, optimizer_config.beta2),
                eps=optimizer_config.eps,
            )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = optimizer_config.learning_rate

        lr_sched_config = self.config.settings.lr_scheduler
        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            last_epoch=self.last_lr_step,
            base_lr=lr_sched_config.base_lr,
            max_lr=optimizer_config.learning_rate,
            warmup_no_steps=lr_sched_config.warmup_no_steps,
            start_decay_after_n_steps=lr_sched_config.start_decay_after_n_steps,
            decay_factor=lr_sched_config.decay_factor,
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
        confidence_scores["global_predicted_distance_error"] = (
            compute_global_predicted_distance_error(
                pde=confidence_scores["predicted_distance_error"],
                distogram_probs=torch.softmax(outputs["distogram_logits"], dim=-1),
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

    def predict_step(self, batch, batch_idx):
        # At the start of inference, load the EMA weights
        if self.cached_weights is None:
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling
            # load_state_dict().
            def clone_param(t):
                return t.detach().clone()

            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])

        query_id = batch["query_id"]

        # Convert seeds back to list
        seed = batch["seed"].cpu().tolist()
        batch["seed"] = seed

        self.reseed(seed[0])  # TODO: assuming we have bs = 1 for now

        # Probably need to change the logic
        logger.debug(
            f"Started inference for {', '.join(query_id)} on rank {self.global_rank} "
            f"step {self.global_step}"
        )
        try:
            batch, outputs = self(batch)

            # Generate confidence scores
            confidence_scores = self._compute_confidence_scores(batch, outputs)
            outputs["confidence_scores"] = confidence_scores

            return batch, outputs

        except torch.OutOfMemoryError as e:
            logger.error(
                f"OOM for query_id(s) {', '.join(query_id)}. "
                f"See {self.output_dir}/predict_err_rank{self.global_rank}.log "
                f"for details."
            )

            self._log_predict_exception(e, query_id)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return None

        except Exception as e:
            logger.error(
                f"Failed for query_id(s) {', '.join(query_id)}: {e}. "
                f"See {self.output_dir}/predict_err_rank{self.global_rank}.log "
                f"for details."
            )

            self._log_predict_exception(e, query_id)

            return None

    def _log_predict_exception(self, e, query_id):
        """Formats and appends exceptions to a rank-specific error log."""

        # Output dir is not specified
        if self.output_dir is None:
            return

        log_file = self.output_dir / f"predict_err_rank{self.global_rank}.log"

        # Get traceback and format message
        error_traceback = traceback.format_exc()

        log_entry = f"""
        ==================================================
        Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Query ID(s): {", ".join(query_id)}
        Error Type: {type(e).__name__}
        Error Message: {e}
        --------------------------------------------------
        Traceback:
        {error_traceback}
        ==================================================
        """

        # Append the entry to the log file
        with open(log_file, "a") as f:
            f.write(log_entry)
