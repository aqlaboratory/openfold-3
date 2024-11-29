import importlib
from pathlib import Path

import torch

from openfold3.core.loss.loss_module import AlphaFold3Loss
from openfold3.core.metrics.confidence import (
    compute_plddt,
    compute_predicted_aligned_error,
    compute_predicted_distance_error,
    compute_weighted_ptm,
)
from openfold3.core.metrics.validation_all_atom import (
    get_validation_metrics,
)
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_frame_atoms,
)
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.af3_all_atom.config.base_config import project_config
from openfold3.projects.af3_all_atom.config.dataset_config_builder import (
    AF3DatasetConfigBuilder,
)
from openfold3.projects.af3_all_atom.model import AlphaFold3
from openfold3.projects.registry import register_project

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    from deepspeed.ops.adam import DeepSpeedCPUAdam

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

    def training_step(self, batch, batch_idx):
        example_feat = next(
            iter(v for v in batch.values() if isinstance(v, torch.Tensor))
        )
        if self.ema.device != example_feat.device:
            self.ema.to(example_feat.device)

        # Run the model
        batch, outputs = self.model(batch)

        # Compute loss
        loss, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

        # Log it
        self._log(loss_breakdown, batch, outputs)

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

        # Run the model
        batch, outputs = self(batch)

        # Compute loss and other metrics
        _, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

        self._log(loss_breakdown, batch, outputs, train=False)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # This is to avoid slow loading for nested dicts in PL
        # Less frequent hanging when non_blocking=True on H200
        # TODO: Determine if this is really needed given other
        #  recent hanging fixes
        def to_device(t):
            return t.to(device=device, non_blocking=True)

        batch = tensor_tree_map(to_device, batch)
        return batch

    def on_train_epoch_start(self):
        # At the start of each virtual epoch we want to resample the set of
        # datapoints to train on
        self.trainer.train_dataloader.dataset.resample_epoch()

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

    def _compute_validation_metrics(
        self, batch, outputs, superimposition_metrics=False
    ) -> dict[str, torch.Tensor]:
        # Computes validation metrics
        metrics = get_validation_metrics(batch, outputs, superimposition_metrics)

        return metrics

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
