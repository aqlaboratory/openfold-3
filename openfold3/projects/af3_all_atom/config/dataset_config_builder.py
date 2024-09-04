import ml_collections as mlc

from openfold3.core.config.dataset_config_builder import DefaultDatasetConfigBuilder


class AF3DatasetConfigBuilder(DefaultDatasetConfigBuilder):
    def __init__(self, project_config: mlc.ConfigDict):
        super().__init__(project_config)
        self.project_loss_config = project_config.extra_configs.loss_weight_modes

    def _update_loss_weight_settings(self, loss_weight_mode: str):
        """Updates datsaset config loss weights section based on selected mode."""
        loss_weight_cfg = self.project_loss_config.default.copy_and_resolve_references()

        allowed_modes = list(self.project_loss_config.custom.keys()) + ["default"]
        if loss_weight_mode not in allowed_modes:
            raise KeyError(
                f"{loss_weight_mode} is not supported, allowed loss modes are: {allowed_modes}"
            )
        elif loss_weight_mode != "default":
            custom_update = self.project_loss_config.custom.get(loss_weight_mode)
            loss_weight_cfg.update(custom_update)

        self.config.config.loss.loss_weights = loss_weight_cfg

    def _update_config(self, config_update):
        super()._update_config(config_update)
        self._update_loss_weight_settings(self.config.config.loss_weight_mode)
