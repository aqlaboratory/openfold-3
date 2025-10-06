import random
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Optional

from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from pydantic import BaseModel, field_validator, model_validator
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.core.data.tools.colabfold_msa_server import MsaComputationSettings
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetConfigKwargs,
    TrainingDatasetPaths,
)
from openfold3.projects.of3_all_atom.project_entry import ModelUpdate

ValidModeType = Literal["train", "predict", "eval", "test"]


class CheckpointConfig(BaseModel):
    """Settings for training checkpoint writing."""

    every_n_epochs: int = 1
    auto_insert_metric_name: bool = False
    save_last: bool = True
    save_top_k: int = -1


class WandbConfig(BaseModel):
    """Configuration for Weights and Biases experiment result logging."""

    project: str
    experiment_name: str
    entity: Optional[str] = None
    group: Optional[str] = None
    id: Optional[str] = None
    offline: bool = False


class LoggingConfig(BaseModel):
    """Settings for training logging."""

    log_lr: bool = True
    log_grads: bool = False
    log_level: Literal["debug", "info", "warning", "error"] | None = None
    wandb_config: WandbConfig | None = None


class DataModuleArgs(BaseModel):
    """Settings for openfold3.core.data.framework.data_module"""

    model_config = PydanticConfigDict(extra="forbid")
    batch_size: int = 1
    data_seed: Optional[int] = None
    num_workers: int = 10
    num_workers_validation: int = 4
    epoch_len: int = 4


class PlTrainerArgs(BaseModel):
    """Arguments to configure pl.Trainer, including settings for number of devices."""

    model_config = PydanticConfigDict(extra="allow")
    max_epochs: int = 1000  # pl_trainer default
    accelerator: str = "gpu"
    precision: int | str = "32-true"
    num_nodes: int = 1
    devices: int = 1  # number of GPUs per node
    profiler: Optional[str] = None
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True
    enable_model_summary: bool = False

    # Extra arguments that are not passed directly to pl.Trainer
    deepspeed_config_path: Path | None = None
    distributed_timeout: Optional[timedelta] = default_pg_timeout
    mpi_plugin: bool = False


class OutputWritingSettings(BaseModel):
    """File formats to use for writing inference prediction results.

    Used by OF3OutputWriter in openfold3.core.runners.writer
    """

    structure_format: Literal["pdb", "cif"] = "cif"
    full_confidence_output_format: Literal["json", "npz"] = "json"
    write_features: bool = False
    write_latent_outputs: bool = False


class ExperimentSettings(BaseModel):
    """General settings for all experiments"""

    mode: ValidModeType
    output_dir: Path = Path("./")
    log_dir: Path | None = None

    @model_validator(mode="after")
    def create_output_dir(cls, model):
        if not model.output_dir.exists():
            model.output_dir.mkdir(parents=True, exist_ok=True)
        return model


class TrainingExperimentSettings(ExperimentSettings):
    """General settings specific for training experiments"""

    mode: ValidModeType = "train"
    seed: int = 42
    restart_checkpoint_path: Optional[str] = None
    load_ema_weights: bool = False
    reset_scheduler: bool = False
    reset_optimizer_states: bool = False

    @classmethod
    @field_validator("restart_checkpoint_path", mode="after")
    def validate_checkpoint_path(cls, ckpt_path: Optional[str]) -> Optional[str]:
        """
        Validates the restart_checkpoint_path.

        The path can be one of the following:
        - None (if no checkpoint is provided).
        - A special string: "last", "hpc", "registry" accepted by PL.
        - A string representing a valid path to a file.
        - A string representing a valid path to a directory (for deepspeed checkpoints).
        """
        if ckpt_path is None:
            return ckpt_path

        # PL accepted strings
        allowed_strings = ["last", "hpc", "registry"]
        if ckpt_path in allowed_strings:
            return ckpt_path

        # The path points to a valid file or directory
        if Path(ckpt_path).exists():
            return ckpt_path

        raise ValueError(
            f'"{ckpt_path}" is not a valid file, directory, or accepted keyword '
            f"({', '.join(allowed_strings)})"
        )


def generate_seeds(start_seed, num_seeds):
    """Helper function for generating random seeds."""
    random.seed(start_seed)
    return [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]


class InferenceExperimentSettings(ExperimentSettings):
    """General settings specific for inference experiments"""

    mode: ValidModeType = "predict"
    seeds: int | list[int] = [42]
    num_seeds: int | None = None
    use_msa_server: bool = False
    use_templates: bool = False

    @model_validator(mode="after")
    def generate_seeds(cls, model):
        """Creates a list of seeds if a list of seeds is not provided."""
        if isinstance(model.seeds, list):
            pass
        elif isinstance(model.seeds, int):
            if model.num_seeds is None:
                raise ValueError(
                    "num_seeds must be provided when seeds is a single int"
                )
            model.seeds = generate_seeds(model.seeds, model.num_seeds)
        elif model.seeds is None:
            raise ValueError("seeds must be provided (either int or list[int])")

        return model


class ExperimentConfig(BaseModel):
    """Base set of arguments expected for all experiments"""

    experiment_settings: ExperimentSettings
    pl_trainer_args: PlTrainerArgs = PlTrainerArgs()
    model_update: ModelUpdate


class TrainingExperimentConfig(ExperimentConfig):
    """Training experiment config"""

    # pydantic model setting to prevent extra fields in main experiment config
    model_config = PydanticConfigDict(extra="forbid")
    # required arguments for training experiment
    dataset_paths: dict[str, TrainingDatasetPaths]
    dataset_configs: dict[str, Any]

    experiment_settings: TrainingExperimentSettings = TrainingExperimentSettings()
    logging_config: LoggingConfig = LoggingConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    model_update: ModelUpdate = ModelUpdate(presets=["train"])
    data_module_args: DataModuleArgs = DataModuleArgs()

    @model_validator(mode="after")
    def synchronize_seeds(cls, model):
        """
        Ensures data_seed in DataModuleArgs is set. If it isn't, it will
        default to the model seed.
        """
        model_seed = model.experiment_settings.seed
        data_seed = model.data_module_args.data_seed

        if data_seed is None:
            model.data_module_args.data_seed = model_seed

        return model


class InferenceExperimentConfig(ExperimentConfig):
    """Inference experiment config"""

    # pydantic model setting to prevent extra fields in main experiment config
    model_config = PydanticConfigDict(extra="forbid")
    # Required inputs for performing inference
    inference_ckpt_path: Path

    experiment_settings: InferenceExperimentSettings = InferenceExperimentSettings()
    model_update: ModelUpdate = ModelUpdate(presets=["predict"])
    data_module_args: DataModuleArgs = DataModuleArgs()
    dataset_config_kwargs: InferenceDatasetConfigKwargs = InferenceDatasetConfigKwargs()
    output_writer_settings: OutputWritingSettings = OutputWritingSettings()
    msa_computation_settings: MsaComputationSettings = MsaComputationSettings()
    template_preprocessor_settings: TemplatePreprocessorSettings = (
        TemplatePreprocessorSettings(mode="predict")
    )

    @model_validator(mode="after")
    def synchronize_seeds(cls, model):
        """
        Ensures data_seed in DataModuleArgs is set. If it isn't, it will
        default to the first model seed in the provided list.
        """
        model_seeds = model.experiment_settings.seeds
        data_seed = model.data_module_args.data_seed

        if data_seed is None:
            model.data_module_args.data_seed = model_seeds[0]

        return model

    @model_validator(mode="after")
    def copy_ccd_file_path(cls, model):
        """Copies ccd_file_path dataset_config_kwargs>template_preprocessor_settings."""
        if model.dataset_config_kwargs.ccd_file_path is not None:
            if model.template_preprocessor_settings.ccd_file_path is not None:
                warnings.warn(
                    "Overwriting ccd_file_path in template_preprocessor_settings with "
                    "dataset_config_kwargs.ccd_file_path. We recommend specifying"
                    "ccd_file_path only in dataset_config_kwargs.",
                    stacklevel=2,
                )
            model.template_preprocessor_settings.ccd_file_path = (
                model.dataset_config_kwargs.ccd_file_path
            )

        return model
