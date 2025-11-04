# Copyright 2025 AlQuraishi Laboratory
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

import copy
import importlib.resources
import logging
from dataclasses import dataclass

from ml_collections import ConfigDict
from pydantic import BaseModel
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.config.config_utils import load_yaml
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom


class ModelUpdate(BaseModel):
    model_config = PydanticConfigDict(extra="forbid")
    presets: list[str] = []
    custom: dict = {}


@dataclass
class OF3ProjectEntry:
    name = "of3_all_atom"
    model_config_base = model_config
    runner = OpenFold3AllAtom
    model_preset_yaml = (
        importlib.resources.files("openfold3.projects.of3_all_atom.config")
        / "model_setting_presets.yml"
    )

    def __post_init__(self):
        with importlib.resources.as_file(self.model_preset_yaml) as preset_path:
            preset_dict = load_yaml(preset_path)

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
        presets: list[str] | None = None,
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

    def validate_model_config(self, model_config: ConfigDict) -> ConfigDict:
        msa_embedder_config = model_config.architecture.msa.msa_module_embedder
        assert not (
            msa_embedder_config.subsample_main_msa
            and msa_embedder_config.subsample_all_msa
        ), (
            "Invalid configuration: both `subsample_main_msa` and `subsample_all_msa` "
            "are set to True. At most one subsampling strategy can be enabled."
        )

    def get_model_config_with_update(
        self, model_update: ModelUpdate | None = None
    ) -> ConfigDict:
        """Returns a model config with updates applied."""
        model_config = self.get_model_config_with_presets(model_update.presets)
        model_config.update(model_update.custom)
        self.validate_model_config(model_config)

        return model_config
