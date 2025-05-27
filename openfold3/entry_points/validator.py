import random
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, model_validator
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.config.config_utils import FilePathOrNone
from openfold3.projects.af3_all_atom.config.dataset_configs import (
    InferenceDatasetConfigKwargs,
    TrainingDatasetPaths,
)
from openfold3.projects.af3_all_atom.project_entry import ModelUpdate


class CheckpointConfig(BaseModel):
    every_n_epochs: int = 1
    auto_insert_metric_name: bool = False
    save_last: bool = True
    save_top_k: int = -1


class WandbConfig(BaseModel):
    project: str = "my project"
    experiment_name: str = "expt_name"
    entity: Optional[str] = None
    group: Optional[str] = None
    id: Optional[str] = None
    offline: bool = False


class LoggingConfig(BaseModel):
    log_lr: bool = True
    log_level: Literal["debug", "info", "warning", "error"] | None = None
    wandb_config: WandbConfig | None = None


class DataModuleArgs(BaseModel):
    """Settings for openfold3.core.data.framework.data_module"""

    model_config = PydanticConfigDict(extra="forbid")
    batch_size: int = 1
    data_seed: int = 1234
    num_workers: int = 10
    epoch_len: int = 4
    num_epochs: int = 1000


class PlTrainerArgs(BaseModel):
    """Arguments to configure pl.Trainer"""

    max_epochs: int = 1000  # pl_trainer default
    accelerator: str = "gpu"
    precision: str = "bf16-mixed"
    num_nodes: int = 1
    devices: int = 1  # number of GPUs per node
    profiler: Optional[str] = None
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True
    enable_model_summary: bool = False

    # Extra arguments that are not passed directly to pl.Trainer
    deepspeed_config_path: Path | None = None
    mpi_plugin: bool = False


class ExperimentSettings(BaseModel):
    """General settings for all experiments"""

    mode: Literal["train", "predict"]
    output_dir: Path


class TrainingExperimentSettings(ExperimentSettings):
    """General settings specific for training experiments"""

    mode: Literal["train", "predict"] = "train"
    seed: int = 42
    restart_checkpoint_path: FilePathOrNone = None
    output_dir: Path = Path("./train_output")


class InferenceExperimentSettings(ExperimentSettings):
    """General settings specific for training experiments"""

    mode: Literal["train", "predict"] = "predict"
    query_json: Path
    inference_ckpt_path: Path
    seeds: int | list[int] = [42]
    num_seeds: int | None = None
    output_dir: Path = Path("./inference_output")

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
            random.seed(model.seeds)
            model.seeds = [random.randint(0, 2**32 - 1) for _ in range(model.num_seeds)]
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

    # required arguments for training experiment
    dataset_paths: dict[str, TrainingDatasetPaths]
    dataset_configs: dict[str, Any]

    experiment_settings: TrainingExperimentSettings = TrainingExperimentSettings()
    logging_config: LoggingConfig = LoggingConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    model_update: ModelUpdate = ModelUpdate(presets=["train"])
    data_module_args: DataModuleArgs = DataModuleArgs()


class InferenceExperimentConfig(ExperimentConfig):
    """Inference experiment config"""

    # TODO: Add MSA configuration settings
    experiment_settings: InferenceExperimentSettings
    model_update: ModelUpdate = ModelUpdate(presets=["predict"])
    data_module_args: DataModuleArgs = DataModuleArgs()
    dataset_config_kwargs: InferenceDatasetConfigKwargs = InferenceDatasetConfigKwargs()
