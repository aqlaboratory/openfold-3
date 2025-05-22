from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.config.path_definitions import FilePathOrNone
from openfold3.projects.af3_all_atom.config.dataset_configs import (
    TrainingDatasetPaths,
)
from openfold3.projects.af3_all_atom.project_entry import ModelUpdate


class CheckpointConfig(BaseModel):
    every_n_epochs: int = 1
    auto_insert_metric_name: bool = False
    save_last: bool = True
    save_top_k: int = -1


class WandbConfig(BaseModel):
    project: str
    experiment_name: str
    entity: Optional[str] = None
    group: Optional[str] = None
    id: Optional[str] = None
    offline: bool = False


class LoggingConfig(BaseModel):
    log_lr: bool = True
    log_level: str | None = None
    checkpoint_config: CheckpointConfig | None = None
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
    """Arguments for openfold3.core.pl_trainer.Trainer"""

    max_epochs: int = 1000  # pl_trainer default
    accelerator: str = "gpu"
    precision: str = "bf16-mixed"
    num_nodes: int = 1
    profiler: Optional[str] = None
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True
    enable_model_summary: bool = False


class ExperimentConfig(BaseModel):
    mode: str
    output_dir: Path
    mpi_plugin: bool = False
    deepspeed_config_path: str | None = None
    compile: bool = False
    num_gpus: int = 1

    pl_trainer_args: PlTrainerArgs
    model_update: ModelUpdate


class TrainingExperimentConfig(ExperimentConfig):
    seed: int = 42
    data_seed: int = 1234
    num_workers: int = 0
    epoch_len: int = 2
    restart_checkpoint_path: FilePathOrNone = None

    logging_config: LoggingConfig = LoggingConfig()
    model_update: ModelUpdate = ModelUpdate(presets=["train"])
    dataset_paths: dict[str, TrainingDatasetPaths]
    dataset_configs: dict[str, Any]
    data_module_args: DataModuleArgs = DataModuleArgs()
