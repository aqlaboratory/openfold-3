import copy
import importlib.resources
import logging
from dataclasses import dataclass
from typing import Optional

from ml_collections import ConfigDict
from pydantic import BaseModel

from openfold3.core.config.config_utils import load_yaml
from openfold3.projects.af3_all_atom.config.model_config import model_config
from openfold3.projects.af3_all_atom.runner import AlphaFold3AllAtom


class ModelUpdate(BaseModel):
    presets: list[str] = []
    custom: dict = {}
    compile: bool = False


@dataclass
class AF3ProjectEntry:
    name = "af3_all_atom"
    model_config_base = model_config
    runner = AlphaFold3AllAtom
    model_preset_yaml = (
        importlib.resources.files("openfold3.projects.af3_all_atom.config")
        / "model_setting_presets.yml"
    )

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

    def get_model_config_with_presets(
        self,
        presets: Optional[list[str]] = None,
    ) -> ConfigDict:
        """Retrieves a config with specified preset applied"""
        config = copy.deepcopy(self.model_config_base)
        config.lock()
        if not presets:
            logging.info(f"Using default model settings for {self.name}")
        else:
            for preset in presets:
                config = self.update_config_with_preset(config, preset)
        return config

    def get_model_config_with_update(
        self, model_update: Optional[ModelUpdate] = None
    ) -> ConfigDict:
        """Returns a model config with updates applied."""
        model_config = self.get_model_config_with_presets(model_update.presets)
        try:
            model_config.update(model_update.custom)
        except ValueError as e:
            # Handle case where the argument is passed as a flattened dict
            # N.B. if the update is a mixture of flattened key and non-flattened dicts,
            #   the non-flattened dict may override the whole ConfigDict
            if "dots in field names" in str(e):
                model_config.update_from_flattened_dict(model_update.custom)
            else:
                raise e

        return model_config
