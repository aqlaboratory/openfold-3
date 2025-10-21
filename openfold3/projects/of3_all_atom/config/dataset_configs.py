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

"""
Dataset configuration for all atom project.

Each dataset has a base DatasetConfig section, which includes:
- dataset_cache_path (train) or inference_query_set (inference): Collection or Path to
    a collection containing the dataset's contents
- Settings for the pytorch.Dataset construction
    e.g. MSASettings, TemplateSettings
    For training datasets, this also includes LossConfig and CropSettings.

The DatasetConfig is wrapped in a DatasetSpec model, which contains additional fields:
    `name`, `dataset_class`, `mode`, and `weight`.
These fields are parsed by the DataModule to create the appropriate Dataset class.

"""

from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    SerializeAsAny,
    model_validator,
)
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.config.config_utils import DirectoryPathOrNone, FilePathOrNone
from openfold3.core.data.framework.data_module import DatasetMode, DatasetSpec
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    CropSettings,
    LossConfig,
    MSASettings,
    TemplateSettings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)


class TrainingDatasetPaths(BaseModel):
    """Dataset paths used by each dataset."""

    dataset_cache_file: FilePath | DirectoryPath
    alignments_directory: DirectoryPathOrNone = None
    alignment_db_directory: DirectoryPathOrNone = None
    alignment_array_directory: DirectoryPathOrNone = None
    target_structures_directory: DirectoryPath
    target_structure_file_format: str
    reference_molecule_directory: DirectoryPath
    template_cache_directory: DirectoryPathOrNone = None
    template_structures_directory: DirectoryPathOrNone = None
    template_structure_array_directory: DirectoryPathOrNone = None
    template_file_format: str | None = None
    ccd_file: FilePathOrNone = None
    use_roda_monomer_format: bool = False

    @model_validator(mode="after")
    def _validate_paths(self):
        def _validate_exactly_one_path_exists(
            group_name: str, path_values: list[Path | None]
        ):
            which_paths_exist = [p is not None for p in path_values]
            if sum(which_paths_exist) != 1:
                existing_paths = [
                    p for p, b in zip(path_values, which_paths_exist, strict=True) if b
                ]
                raise ValueError(
                    f"Exactly one path in set of {group_name} should exist."
                    f"Found {existing_paths} exist."
                )

        _validate_exactly_one_path_exists(
            "alignment paths",
            [
                self.alignments_directory,
                self.alignment_db_directory,
                self.alignment_array_directory,
            ],
        )
        _validate_exactly_one_path_exists(
            "template_paths",
            [
                self.template_structures_directory,
                self.template_structure_array_directory,
            ],
        )
        return self


class DefaultDatasetConfigSection(BaseModel):
    """Base configuration settings for all atom datasets.

    Datasets for this project are defined in
      `openfold3.core.data.framework.single_datasets`

    This BaseModel only defines the "config" section for the dataset inputs.
    The full dataset class specification is provided in TrainingDatasetSpec,
      and contains this BaseModel as a section.

    A separate subclass is created for each dataset type below and
    added to the DatasetConfigRegistry.
        - WeightedPDBConfig
        - ProteinMonomerDistillationConfig
        - DisorderedPDBConfig
        - ValidationPDBConfig
    """

    model_config = PydanticConfigDict(extra="forbid")
    name: str
    debug_mode: bool = False
    sample_in_order: bool = False
    dataset_paths: TrainingDatasetPaths
    msa: MSASettings = MSASettings()
    template: TemplateSettings = TemplateSettings()
    loss: LossConfig = LossConfig()


class DatasetConfigRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, config: DefaultDatasetConfigSection) -> None:
        cls._registry[name] = config

    @classmethod
    def get(cls, name: str) -> DefaultDatasetConfigSection:
        config_class = cls._registry.get(name)
        if not config_class:
            raise ValueError(
                f"{name} was not found in the dataset registry,"
                f"available config types are are {cls._registry.keys()}"
            )
        return config_class


DATASET_CONFIG_REGISTRY = DatasetConfigRegistry()


def register_dataset_config(name: str) -> None:
    """Helper decorator function to label datasets."""

    def _decorator(config_class):
        DATASET_CONFIG_REGISTRY.register(name=name, config=config_class)

    return _decorator


### Configuration defaults for each dataset class


@register_dataset_config("WeightedPDBDataset")
class WeightedPDBConfig(DefaultDatasetConfigSection):
    crop: CropSettings = CropSettings()
    sample_weights: dict = {
        "a_prot": 3.0,
        "a_nuc": 3.0,
        "a_ligand": 1.0,
        "w_chain": 0.5,
        "w_interface": 1.0,
    }


@register_dataset_config("ProteinMonomerDataset")
class ProteinMonomerConfig(DefaultDatasetConfigSection):
    sample_in_order: bool = True
    crop: CropSettings = CropSettings(
        crop_weights={
            "contiguous": 0.25,
            "spatial": 0.75,
            "spatial_interface": 0.0,
        }
    )
    loss: LossConfig = LossConfig(
        loss_weights={
            "bond": 0.0,
            "smooth_lddt": 4.0,
            "mse": 4.0,
            "distogram": 3e-2,
            # These losses are zero for the protein_monomer_distillation set
            "experimentally_resolved": 0.0,
            "plddt": 0.0,
            "pae": 0.0,
            "pde": 0.0,
        }
    )


@register_dataset_config("DisorderedPDBDataset")
class DisorderedPDBConfig(DefaultDatasetConfigSection):
    sample_weights: dict = {
        "a_prot": 3.0,
        "a_nuc": 3.0,
        "a_ligand": 1.0,
        "w_chain": 0.5,
        "w_interface": 1.0,
    }
    crop: CropSettings = CropSettings()
    disable_non_protein_diffusion_weights: bool = True
    loss: LossConfig = LossConfig(
        loss_weights={
            "bond": 0.0,
            "smooth_lddt": 4.0,
            "mse": 4.0,
            "distogram": 3e-2,
            # These losses are zero for distillation sets
            "experimentally_resolved": 0.0,
            "plddt": 0.0,
            "pae": 0.0,
            "pde": 0.0,
        }
    )


@register_dataset_config("ValidationPDBDataset")
class ValidationPDBConfig(DefaultDatasetConfigSection):
    template: TemplateSettings = TemplateSettings(take_top_k=True)


class TrainingDatasetSpec(DatasetSpec):
    """Full dataset specification for all atom style projects.

    A list of these configurations can be provided to
    `core.data.framework.data_module` to create
    `torch.Datasets` needed for all atom training.

    The correct DatasetConfig to use for each dataset will be inferred
      from the `dataset_class` argument.
    """

    name: str
    dataset_class: str
    mode: DatasetMode
    weight: float | None = None
    config: SerializeAsAny[BaseModel] = Field(
        default_factory=lambda: DefaultDatasetConfigSection
    )

    @model_validator(mode="before")
    def load_config(cls, values: dict[str, Any]):
        dataset_class = values.get("dataset_class")
        config_class = DatasetConfigRegistry.get(dataset_class)
        config_data = values.get("config", {})
        config_data["name"] = values.get("name")
        values["mode"] = DatasetMode(values.get("mode"))

        values["config"] = config_class.model_validate(config_data)
        return values


class InferenceDatasetConfigKwargs(BaseModel):
    """Class to hold msa and template kwargs for inference pipeline"""

    ccd_file_path: FilePathOrNone = None
    msa: MSASettings = MSASettings()
    template: TemplateSettings = TemplateSettings(take_top_k=True)


class InferenceJobConfig(BaseModel):
    """Configuration section for Inference Datasets"""

    query_set: InferenceQuerySet
    seeds: list[int] = [42]
    ccd_file_path: FilePathOrNone = None
    msa: MSASettings = MSASettings()
    template: TemplateSettings = TemplateSettings()
    template_preprocessor_settings: TemplatePreprocessorSettings


class InferenceDatasetSpec(DatasetSpec):
    """Full specification for inference dataset to be passed into DataModule"""

    name: str = "inference"
    dataset_class: str = "InferenceDataset"
    mode: DatasetMode = DatasetMode.prediction
    weight: float | None = None
    config: InferenceJobConfig
