# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import importlib
import logging
import traceback
import warnings
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from torchmetrics import MeanMetric, MetricCollection, PearsonCorrCoef

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.metrics.aggregate_confidence_ranking import get_confidence_scores
from openfold3.core.metrics.model_selection import (
    compute_final_model_selection_metric,
    compute_valid_model_selection_metrics,
)
from openfold3.core.metrics.quality import (
    get_metrics,
    get_metrics_chunked,
)
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.grad_manager import PerSampleGradManager, compute_global_norm
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.core.utils.timing import PerformanceTimer
from openfold3.projects.of3_all_atom.config.model_config import (
    model_selection_metric_weights_config,
)
from openfold3.projects.of3_all_atom.constants import (
    CORRELATION_METRICS,
    METRIC_DENOMINATOR_ATTRS,
    TRAIN_LOGGED_METRICS,
    TRAIN_LOSSES,
    VAL_LOGGED_METRICS,
    VAL_LOSSES,
)
from openfold3.projects.of3_all_atom.model import OpenFold3

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed

logger = logging.getLogger(__name__)
# We define extra metrics that will cause this warning depending on the training stage
# Only metrics with values present are logged, so we can ignore this error
warnings.filterwarnings(
    "ignore",
    message=(
        r"The ``compute`` method of metric .* was called before the ``update`` method"
    ),
    category=UserWarning,
    module="torchmetrics",
)

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


class OpenFold3AllAtom(ModelRunner):
    def __init__(self, model_config, log_dir: Path = None):
        super().__init__(model_class=OpenFold3, config=model_config)

        self.log_dir = log_dir

        self.loss = OpenFold3Loss(config=model_config.architecture.loss_module)

        self.model_selection_weights = model_selection_metric_weights_config[
            self.config.settings.model_selection_weight_scheme
        ]

        # Settings for per-sample gradient clipping
        self.per_sample_grad_clipping = (
            model_config.settings.gradient_clipping.per_sample_clipping
        )
        self.grad_manager = None
        if self.per_sample_grad_clipping:
            self.grad_manager = PerSampleGradManager(
                gradient_clip_val=model_config.settings.gradient_clipping.clip_val,
                accumulate_grad_batches=model_config.settings.manual_optimization.accumulate_grad_batches,
                log_grad_norm=model_config.settings.debug.log_grad_norm,
            )
            self.automatic_optimization = False
            self.log_lr = model_config.settings.manual_optimization.log_lr

    @property
    def version(self):
        v = self.model.version_tensor.long().tolist()
        return f"{v[0]}.{v[1]}.{v[2]}"

    def setup(self, stage: str):
        # Setup metrics
        self._setup_train_metrics()
        self._setup_val_metrics()

        # Keep grads enabled for confidence head parameters only
        if stage == "fit" and self.config.settings.train_confidence_only:
            exempt_submodule = [
                self.model.aux_heads.pairformer_embedding,
                self.model.aux_heads.pde,
                self.model.aux_heads.plddt,
                self.model.aux_heads.experimentally_resolved,
                self.model.aux_heads.pae,
            ]
            self._freeze_model_params(exempt_submodule=exempt_submodule)

        # Initialize the gradient manager if doing per-sample grad clipping
        if self.per_sample_grad_clipping:
            self.grad_manager.setup(
                model=self.model, trainer=self.trainer, logger=self.logger
            )
            self._identify_confidence_params()

    def _freeze_model_params(self, exempt_submodule: list[torch.nn.Module]):
        """Freeze all model parameters excluding those specified in exempt_submodule."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only the exempt parameters
        for layer in exempt_submodule:
            for param in layer.parameters():
                param.requires_grad = True

    def _identify_confidence_params(self):
        """
        Identifies which parameters belong to confidence heads.
        """
        confidence_modules_prefixes = [
            "aux_heads.pairformer_embedding",
            "aux_heads.pde",
            "aux_heads.plddt",
            "aux_heads.experimentally_resolved",
            "aux_heads.pae",
        ]

        self.confidence_param_names = set()

        for name, _ in self.model.named_parameters():
            # Check if this param belongs to confidence module
            is_confidence = any(
                name.startswith(f"{prefix}.") for prefix in confidence_modules_prefixes
            )
            if is_confidence:
                self.confidence_param_names.add(name)

    def reseed(self, seed):
        pl.seed_everything(seed)

    def _setup_train_metrics(self):
        """Set up training loss and metric collection objects."""

        # TODO: Forcing naming convention to be compatible with older runs
        #  Make consistent later
        # Initialize all training epoch metric objects
        train_losses = {
            loss_name: MeanMetric(nan_strategy="warn", sync_on_compute=False)
            for loss_name in TRAIN_LOSSES
        }
        self.train_losses = MetricCollection(
            train_losses, prefix="train/", postfix="_epoch"
        )

        train_metrics = {
            metric_name: MeanMetric(nan_strategy="warn", sync_on_compute=False)
            for metric_name in TRAIN_LOGGED_METRICS
        }

        self.train_metrics = MetricCollection(train_metrics, prefix="train/")

    def _setup_val_metrics(self):
        """Set up validation loss and metric collection objects."""

        # Initialize all validation epoch metric objects
        val_losses = {
            loss_name: MeanMetric(nan_strategy="warn", sync_on_compute=False)
            for loss_name in VAL_LOSSES
        }
        self.val_losses = MetricCollection(val_losses, prefix="val/")

        val_metrics = {
            metric_name: MeanMetric(nan_strategy="warn", sync_on_compute=False)
            for metric_name in VAL_LOGGED_METRICS
        }
        val_metrics.update(
            {
                metric_name: PearsonCorrCoef(num_outputs=1, sync_on_compute=False)
                for metric_name in CORRELATION_METRICS
            }
        )
        self.val_metrics = MetricCollection(val_metrics, prefix="val/")

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

        if metric_log_name not in metric_collection.keys():  # noqa: SIM118
            raise ValueError(
                f"Metric {metric_log_name} is not being tracked and will "
                f"not appear in epoch metrics. Please add it to "
                f"the {phase.upper()}_LOSSES or METRICS constants."
            )

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

    def _log(
        self, loss_breakdown, batch, outputs, train=True, log_train_step_metrics=True
    ):
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
            if train and log_train_step_metrics:
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

            # Only log steps for training
            if train and log_train_step_metrics:
                self.log(
                    f"{metric_log_name}_step",
                    metric_value,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )

    def _is_opt_step_ready(self, batch_idx: int) -> bool:
        """
        Checks if the optimizer step should be performed.
        Used in manual mode.
        """
        if self.per_sample_grad_clipping:
            accum_steps = self.grad_manager.accumulate_grad_batches
        else:
            accum_steps = self.trainer.accumulate_grad_batches

        is_last_step_of_cycle = (batch_idx + 1) % accum_steps == 0
        return is_last_step_of_cycle or self.trainer.is_last_batch

    def _get_sample_disabled_param_names(self, loss_weights: dict) -> set | None:
        """
        Returns a list of confidence head parameters that should be disabled
        when counting grads across ranks, else None.
        """
        confidence_loss_name = (
            self.config.architecture.loss_module.confidence_loss_names
        )

        total_conf_weight = sum(
            loss_weights[name].item() for name in confidence_loss_name
        )

        is_valid_confidence_sample = total_conf_weight > 0

        # Confidence losses valid are not valid for distillation samples or
        # samples with resolution out-of-bounds
        if not is_valid_confidence_sample:
            # Return the pre-computed list of confidence param names
            return self.confidence_param_names

        # If no params are disabled, return None
        return None

    def _training_step_manual_clip(self, batch, batch_idx):
        assert len(batch["pdb_id"]) == 1, (
            "Currently only local batch size of 1 per GPU is supported."
        )

        if self.trainer.world_size > 1:
            assert isinstance(self.trainer.strategy, DDPStrategy), (
                "Per-sample gradient clipping is only supported with DDPStrategy."
            )

        example_feat = batch["token_mask"]
        if self.ema.device != example_feat.device:
            self.ema.to(example_feat.device)

        if self.grad_manager.device != example_feat.device:
            self.grad_manager.to(example_feat.device)

        pdb_id = ", ".join(batch["pdb_id"])
        preferred_chain_or_interface = batch["preferred_chain_or_interface"]
        logging_info = {
            "pdb_id": pdb_id,
            "preferred_chain_or_interface": preferred_chain_or_interface,
        }

        logger.debug(
            f"Started model forward pass for {pdb_id} with preferred chain or "
            f"interface {preferred_chain_or_interface} on rank {self.global_rank} "
            f"step {self.global_step}"
        )

        opt = self.optimizers()

        # zero_grad() must be called on every micro-batch to ensure
        # self.manual_backward() sets p.grad instead of adding to it
        # when doing gradient accumulation.
        # It's probably overkill to handle the clipping this exactly
        # instead of just using the averaged microbatch, but I'll revisit
        # that later if needed.
        opt.zero_grad()

        try:
            # Only required when running in distributed mode
            sync_context = (
                self.trainer.model.no_sync()
                if self.trainer.world_size > 1
                else nullcontext()
            )

            # When using DDP, this disables the automatic sync that would happen on
            # manual_backward and break the per-sample grad clipping
            with sync_context:
                # Run the model
                batch, outputs = self.model(batch)

                # Compute loss
                loss, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

                self.manual_backward(loss)

                disabled_params = self._get_sample_disabled_param_names(
                    loss_weights=batch["loss_weights"]
                )
                self.grad_manager.clip_and_accumulate(
                    logging_info=logging_info, disabled_params=disabled_params
                )

            if self._is_opt_step_ready(batch_idx):
                # Average and sync grads
                self.grad_manager.sync_and_average_grads()

                self.grad_manager.log_average_grad_norm()

                opt.step()
                self.lr_schedulers().step()

                # Zero the grad accumulator
                self.grad_manager.reset_accumulator()

                # Log LR and step metrics only after the optimizer step
                # to mimic logging behavior when using automatic optimization
                if self.log_lr:
                    self.log(
                        "AlphaFoldLRScheduler",
                        opt.param_groups[0]["lr"],
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                        sync_dist=False,
                    )

                self._log(
                    loss_breakdown,
                    batch,
                    outputs,
                    train=True,
                    log_train_step_metrics=True,
                )

                # Workaround for PL step logging issues. Avoids using
                # `self.trainer.fit_loop.epoch_loop._batches_that_stepped` if this
                # metric exists.
                self.log(
                    "step",
                    self.global_step,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )

            else:
                # Always update epoch metrics
                self._log(
                    loss_breakdown,
                    batch,
                    outputs,
                    train=True,
                    log_train_step_metrics=False,
                )

        except Exception:
            logger.exception(
                f"Train step failed with pdb id {pdb_id} with "
                f"preferred chain or interface {preferred_chain_or_interface}"
            )

            # Clear grad accumulator on error
            # Only really necessary if trainer is not reinitialized after exception
            self.grad_manager.reset_accumulator()

            raise

        return loss

    def _training_step(self, batch):
        example_feat = batch["token_mask"]
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

            self._log(loss_breakdown, batch, outputs)

        except Exception:
            logger.exception(
                f"Train step failed with pdb id {pdb_id} with "
                f"preferred chain or interface {preferred_chain_or_interface}"
            )
            raise

        return loss

    def training_step(self, batch, batch_idx):
        if self.per_sample_grad_clipping:
            return self._training_step_manual_clip(batch=batch, batch_idx=batch_idx)

        return self._training_step(batch=batch)

    def eval_step(self, batch, batch_idx):
        pdb_id = batch["pdb_id"]
        is_repeated_sample = batch.get("repeated_sample")
        if is_repeated_sample:
            logger.debug(
                f"Skipping repeated sample {', '.join(pdb_id)} on rank "
                f"{self.global_rank}"
            )
            return

        logger.debug(
            f"Started validation for {', '.join(pdb_id)} on rank {self.global_rank} "
            f"step {self.global_step}"
        )

        try:
            # Run the model
            batch, outputs = self(batch)

            # Compute loss and other metrics
            _, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

            self._log(loss_breakdown, batch, outputs, train=False)

        except Exception:
            logger.exception(f"Validation step failed with pdb id {', '.join(pdb_id)}")
            raise

    def on_validation_epoch_start(self):
        # At the start of validation, load the EMA weights

        assert self.cached_weights is None

        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling
        # load_state_dict().
        self.cached_weights = tensor_tree_map(
            lambda t: t.detach().clone(), self.model.state_dict()
        )

        self.model.load_state_dict(self.ema.state_dict()["params"])

    def on_before_optimizer_step(self, *args, **kwargs):
        """
        Logs unclipped grad norm and gradients for the single-transition
        linear_out layers. This logging can be enabled in config.settings.debug.

        These gradients can be associated with instabilities, so we're logging them on
        every single step (bypassing log_every_n_steps) for more accurate monitoring.
        """
        debug_settings = self.config.settings.debug

        # Transition layers included in this logging are frozen when
        # training confidence only
        should_log_extra_metrics = (
            False
            if self.config.settings.train_confidence_only
            else debug_settings.log_extra_grad_metrics
        )
        should_log_grad_norm = debug_settings.log_grad_norm

        if not should_log_extra_metrics and not should_log_grad_norm:
            return

        extra_grad_metrics = {}

        # Only rank zero will actually log the gradients
        log_grad_metrics = self.trainer.is_global_zero and self.logger is not None

        # Only log 4 representative blocks to reduce overhead
        block_idxs = [0, 16, 32, 47]

        # To see if this slows down training, we additionally log runtimes from the
        # global_zero process
        # TODO: Set this to log-level INFO and configure per-module log-levels in a more
        # principled way
        timing_context = partial(PerformanceTimer, logger=logger, level=logging.WARNING)
        log_timing = log_grad_metrics and debug_settings.profile_grad_logging

        context = (
            timing_context("Extra-gradient fetching and calculation")
            if log_timing
            else nullcontext()
        )

        with context:
            if should_log_extra_metrics:
                for idx in block_idxs:
                    block = self.model.pairformer_stack.blocks[idx]
                    param = block.single_transition.linear_out.weight

                    if isinstance(self.trainer.strategy, DeepSpeedStrategy):
                        # Needs to be called on every rank to avoid hanging
                        # https://github.com/deepspeedai/DeepSpeed/issues/7117#issuecomment-2717974187
                        grad = deepspeed.utils.safe_get_full_grad(param)
                    else:
                        grad = param.grad

                    assert not grad.requires_grad

                    if log_grad_metrics:
                        tag = (
                            f"extra_gradients/model.pairformer_stack.blocks.{idx}."
                            "single_transition.linear_out.weight"
                        )

                        extra_grad_metrics[f"{tag}_norm"] = grad.norm().item()
                        extra_grad_metrics[f"{tag}_max"] = grad.abs().max().item()

            if not self.per_sample_grad_clipping and should_log_grad_norm:
                # Compute global grad norm for per-batch grad clipping
                # Per sample clipping handles this logging in the grad manager
                global_norm, _ = compute_global_norm(parameters=self.model.parameters())
                extra_grad_metrics["extra_gradients/avg_unclipped_grad_norm"] = (
                    global_norm.item()
                )

        if log_grad_metrics:
            context = (
                timing_context("Extra-gradient logging")
                if log_timing
                else nullcontext()
            )
            with context:
                # NOTE: This out-of-schedule logging might interact a bit weirdly with
                # the WandB Step, so always plot against trainer/global_step
                self.logger.log_metrics(extra_grad_metrics, step=self.global_step)

    def _log_epoch_metrics(
        self, metrics: MetricCollection, compute_model_selection: bool = False
    ):
        """Log aggregated epoch metrics for training or validation.

        Args:
            metrics: MetricCollection object containing the metrics to log
        """
        if not self.trainer.sanity_checking:
            # Sync and reduce metrics across ranks
            # Done separately from compute() to get the sample counts
            # so that only enabled metrics are logged
            for metric in metrics.values():
                metric.sync()
                metric._should_unsync = False

            metrics_output = metrics.compute()

            # Only log metrics that have been updated
            enabled_metrics = {}
            for name, result in metrics_output.items():
                metric_obj = metrics[name]
                metric_type = type(metric_obj)

                # Get the sample count attribute name (e.g., 'weight')
                attr_name = METRIC_DENOMINATOR_ATTRS.get(metric_type)

                if attr_name is None:
                    raise NotImplementedError(
                        f"Failed to get sample count for metric type "
                        f"'{metric_type.__name__}'. Please add this metric "
                        f"to the METRIC_DENOMINATOR_ATTRS constant."
                    )

                n_samples = getattr(metric_obj, attr_name).sum().item()
                if n_samples > 0:
                    enabled_metrics[name] = result

            if self.per_sample_grad_clipping and self.logger is not None:
                self.logger.log_metrics(enabled_metrics, step=self.global_step)
            else:
                for name, result in enabled_metrics.items():
                    self.log(
                        name,
                        result,
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        sync_dist=False,  # Already synced
                    )

            if compute_model_selection:
                model_selection = compute_final_model_selection_metric(
                    metrics=metrics_output,
                    model_selection_weights=self.model_selection_weights,
                )

                if self.per_sample_grad_clipping and self.logger is not None:
                    self.logger.log_metrics(
                        {"val/model_selection": model_selection}, step=self.global_step
                    )
                else:
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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after optimizer.step(). Gradients are present and clipped."""

        # Skip grad accumulation steps
        if not self._is_opt_step_ready(batch_idx):
            return

        # EMA weight update
        self.ema.update(self.model)

        # Log the clipped step norm when not using per-sample gradient clipping
        # In order to match the logging step of per-sample grad clipping,
        # the step is shifted by 1
        should_log_grad_norm = (
            self.config.settings.debug.log_grad_norm and self.logger is not None
        )
        if (
            not self.per_sample_grad_clipping
            and should_log_grad_norm
            and self.trainer.global_step > 0
        ):
            global_norm, _ = compute_global_norm(parameters=self.model.parameters())
            self.logger.log_metrics(
                {"extra_gradients/avg_clipped_grad_norm": global_norm.item()},
                step=self.global_step - 1,
            )

    def _get_train_sampler(self):
        dl = self.trainer.train_dataloader
        # If PL uses multiple loaders, dl can be CombinedLoader etc.
        # Handle the simple/common case first:
        sampler = getattr(dl, "sampler", None)

        # If it's a BatchSampler, the underlying sampler is sampler.sampler
        if (
            sampler is not None
            and hasattr(sampler, "sampler")
            and (hasattr(sampler, "batch_size") and hasattr(sampler, "drop_last"))
        ):
            # only unwrap if it looks like a BatchSampler wrapper
            # (BatchSampler has .sampler and .batch_size, .drop_last)
            sampler = sampler.sampler
        return sampler

    def on_train_epoch_start(self):
        sampler = self._get_train_sampler()

        logger.info(
            f"Rank - {self.global_rank} starting epoch {self.trainer.current_epoch} "
            f"sampler epoch: {sampler.epoch} "
            f"global_step={self.trainer.global_step} "
            f"next_dataset_indices={self.trainer.datamodule.next_dataset_indices}"
        )

    def on_train_epoch_end(self):
        """Log aggregated epoch metrics for training."""
        self._log_epoch_metrics(metrics=self.train_losses)
        self._log_epoch_metrics(metrics=self.train_metrics)
        sampler = self._get_train_sampler()
        logger.info(
            f"Rank - {self.global_rank} finished epoch {self.trainer.current_epoch} "
            f"sampler epoch: {sampler.epoch} "
            f"global_step={self.trainer.global_step} "
            f"next_dataset_indices={self.trainer.datamodule.next_dataset_indices}"
        )

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
        num_samples = self.config.architecture.shared.diffusion.no_full_rollout_samples
        num_atoms = outputs["atom_positions_predicted"].shape[-2]
        compute_per_sample = (
            num_samples > 1
            and self.config.settings.memory.eval.per_sample_atom_cutoff is not None
            and num_atoms > self.config.settings.memory.eval.per_sample_atom_cutoff
        )

        confidence_scores = get_confidence_scores(
            batch=batch,
            outputs=outputs,
            config=self.config,
            compute_per_sample=compute_per_sample,
        )

        return confidence_scores

    def predict_step(self, batch, batch_idx):
        # Skip if dataloader fails -> returns empty batch
        is_repeated_sample = batch.get("repeated_sample")
        valid_sample = batch.get("valid_sample")
        if not valid_sample or is_repeated_sample:
            return

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
                f"See {self.log_dir}/predict_err_rank{self.global_rank}.log "
                f"for details."
            )

            self._log_predict_exception(e, query_id)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(
                f"Failed for query_id(s) {', '.join(query_id)}: {e}. "
                f"See {self.log_dir}/predict_err_rank{self.global_rank}.log "
                f"for details."
            )

            self._log_predict_exception(e, query_id)

    def _log_predict_exception(self, e, query_id):
        """Formats and appends exceptions to a rank-specific error log."""

        # Output dir is not specified
        if self.log_dir is None:
            return

        log_file = self.log_dir / f"predict_err_rank{self.global_rank}.log"

        # Get traceback and format message
        error_traceback = traceback.format_exc()

        lines = [
            "==================================================",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Query ID(s): {', '.join(query_id)}",
            f"Error Type: {type(e).__name__}",
            f"Error Message: {e}",
            "--------------------------------------------------",
            f"Traceback:{error_traceback}",
            "==================================================",
        ]
        log_entry = "\n".join(lines)

        # Append the entry to the log file
        with open(log_file, "a") as f:
            f.write(log_entry)
