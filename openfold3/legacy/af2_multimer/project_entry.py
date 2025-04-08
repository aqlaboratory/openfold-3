import copy
import logging
from dataclasses import dataclass
from typing import Optional

from ml_collections import ConfigDict

from openfold3.core.config.config_utils import load_yaml
from openfold3.legacy.af2_multimer.config.base_config import config as multimer_config
from openfold3.legacy.af2_multimer.runner import AlphaFoldMultimer


@dataclass
class AF2MultimerProjectEntry:
    name = "af2_multimer"
    model_config_base = multimer_config
    runner = AlphaFoldMultimer
    model_preset_yaml = "openfold3/legacy/af2_multimer/config/reference_config.yml"

    def __post_init__(self):
        preset_dict = load_yaml(self.model_preset_yaml)
        self.model_presets = list(preset_dict.keys())

    def update_config_with_preset(self, config: ConfigDict, preset: str) -> ConfigDict:
        """Updates a given configdict with a preset that is part of the ProjectEntry"""
        if preset not in self.model_presets:
            raise KeyError(
                f"{preset} preset is not supported for {self.name}"
                f"Allowed presets are {self.model_presets}"
            )
        reference_configs = ConfigDict(load_yaml(self.model_preset_yaml))
        preset_model_config = reference_configs[preset]
        config.update(preset_model_config)
        return config

    def get_config_with_presets(
        self,
        presets: Optional[list[str]] = None,
    ) -> ConfigDict:
        """Retrieves a config with specified preset applied"""
        config = copy.deepcopy(self.model_config_base)
        if not presets:
            logging.info(f"Using default model settings for {self.name}")
        else:
            for preset in presets:
                config = self.update_config_with_preset(config, preset)
        return config
