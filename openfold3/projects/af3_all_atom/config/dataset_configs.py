""" """

from pathlib import Path
from typing import Annotated, Any, Optional, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    DirectoryPath,
    Field,
    FilePath,
    SerializeAsAny,
    model_validator,
)
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.data.framework.data_module import DatasetMode
from openfold3.projects.af3_all_atom.config.dataset_config_components import (
    CropSettings,
    LossConfig,
    MSASettings,
    TemplateSettings,
)


def is_path_none(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if isinstance(value, Path):
        return value
    elif value is None or value.lower() in ["none", "null"]:
        return None
    else:
        return Path(value)


FilePathOrNone = Annotated[Optional[FilePath], BeforeValidator(is_path_none)]
DirectoryPathOrNone = Annotated[Optional[DirectoryPath], BeforeValidator(is_path_none)]


class TrainingDatasetPaths(BaseModel):
    """Dataset paths used by each dataset."""

    dataset_cache_file: FilePath
    alignments_directory: DirectoryPathOrNone = None
    alignment_db_directory: DirectoryPathOrNone = None
    alignment_array_directory: DirectoryPathOrNone = None
    target_structures_directory: DirectoryPath
    target_structure_file_format: str
    reference_molecule_directory: DirectoryPath
    template_cache_directory: DirectoryPathOrNone = None
    template_structures_directory: DirectoryPathOrNone = None
    template_structure_array_directory: DirectoryPathOrNone = None
    template_file_format: Optional[str] = None
    ccd_file: FilePathOrNone = None


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
                f"{name} was not found in the dataset registry, available config types are are {cls._registry.keys()}"
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


@register_dataset_config("ProteinMonomerDistillationDataset")
class ProteinMonomerDistillationConfig(DefaultDatasetConfigSection):
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
            "experimentally_resolved": 0.0,
            # These losses are zero for the protein_monomer_distillation set
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


@register_dataset_config("ValidationPDBDataset")
class ValidationPDBConfig(DefaultDatasetConfigSection):
    template: TemplateSettings = TemplateSettings(take_top_k=True)


class TrainingDatasetSpec(BaseModel):
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
    weight: Optional[float] = None
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

        values["config"] = config_class(**config_data)
        return values
