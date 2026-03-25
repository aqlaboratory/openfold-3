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

import warnings
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
from pydantic_core import PydanticUndefined

from openfold3.core.config.config_utils import (
    DirectoryPathOrNone,
    FilePathOrNone,
    deep_update,
)
from openfold3.core.data.framework.data_module import DatasetMode, DatasetSpec
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    ChainCropSettings,
    CropSettings,
    CropWeights,
    LossConfig,
    MSASettings,
    TemplateSettings,
    TokenCropSettings,
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
    def _validate_alignment_paths(self):
        path_values = [
            self.alignments_directory,
            self.alignment_db_directory,
            self.alignment_array_directory,
        ]
        existing_paths = [p for p in path_values if p is not None]
        if len(existing_paths) != 1:
            raise ValueError(
                f"Exactly one path in set of alignment paths should exist."
                f" Found {existing_paths} exist."
            )
        return self

    @model_validator(mode="after")
    def _validate_template_paths(self):
        if (
            self.template_structures_directory is not None
            and self.template_structure_array_directory is not None
        ):
            raise ValueError(
                "Only one template path should be provided. "
                f"Found {self.template_structures_directory} "
                f"and {self.template_structure_array_directory}."
            )

        if (
            self.template_structures_directory is None
            and self.template_structure_array_directory is None
        ):
            warnings.warn(
                "No template paths provided. "
                "Templates will not be used for this dataset.",
                stacklevel=2,
            )

        if (
            self.template_structures_directory is not None
            and self.template_file_format not in ["cif", "pdb"]
        ):
            raise ValueError(
                f"template_file_format must be one of: cif, pdb. "
                f"Got: {self.template_file_format}"
            )

        if (
            self.template_structure_array_directory is not None
            and self.template_file_format not in ["pkl", "npz"]
        ):
            raise ValueError(
                f"template_file_format must be one of: pkl, npz. "
                f"Got: {self.template_file_format}"
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
    crop: CropSettings = CropSettings()
    loss: LossConfig = LossConfig()

    @model_validator(mode="before")
    @classmethod
    def merge_partial_update_with_defaults(cls, data: dict) -> dict:
        """
        Intercepts the user dataset config updates. If a field is provided in `data`
        but the class also has a default set for it, merge the data into the default
        instead of replacing it. This allows partial updates to nested config sections.
        """
        if not isinstance(data, dict):
            return data

        for field_name, field_info in cls.model_fields.items():
            # Apply for nested updates
            if field_name in data and isinstance(data[field_name], dict):
                # Get default value
                default_val = None
                if field_info.default_factory is not None:
                    default_val = field_info.default_factory()
                elif field_info.default is not PydanticUndefined:
                    default_val = field_info.default

                if isinstance(default_val, BaseModel):
                    default_dict = default_val.model_dump()

                    # Merge and update default values with user provided data
                    merged = deep_update(default_dict, data[field_name])
                    data[field_name] = merged

        return data


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
    crop: CropSettings = CropSettings(
        chain_crop=ChainCropSettings(enabled=True),
    )
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
        token_crop=TokenCropSettings(
            crop_weights=CropWeights(
                contiguous=0.25,
                spatial=0.75,
                spatial_interface=0.0,
            )
        )
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


@register_dataset_config("RNAMonomerDataset")
class RNAMonomerConfig(DefaultDatasetConfigSection):
    crop: CropSettings = CropSettings(
        token_crop=TokenCropSettings(
            crop_weights=CropWeights(
                contiguous=0.25,
                spatial=0.75,
                spatial_interface=0.0,
            )
        )
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
    crop: CropSettings = CropSettings(
        chain_crop=ChainCropSettings(enabled=True),
    )
    sample_weights: dict = {
        "a_prot": 3.0,
        "a_nuc": 3.0,
        "a_ligand": 1.0,
        "w_chain": 0.5,
        "w_interface": 1.0,
    }
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
    crop: CropSettings = CropSettings(token_crop=TokenCropSettings(enabled=False))
    msa: MSASettings = MSASettings(subsample_main=False)
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
    msa: MSASettings = MSASettings(subsample_main=False)
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
