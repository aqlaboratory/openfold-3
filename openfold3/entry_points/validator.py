from typing import Any, Optional

from pydantic import BaseModel

from openfold3.core.config.path_definitions import FilePathOrNone
from openfold3.projects.af3_all_atom.config.dataset_configs import (
    InferenceDatasetSpec,
    TrainingDatasetPaths,
    TrainingDatasetSpec,
)
from openfold3.projects.af3_all_atom.project_entry import ModelUpdate


class CheckpointConfig(BaseModel):
    every_n_epochs: int
    auto_insert_metric_name: bool
    save_last: bool
    save_top_k: int


class WandbConfig(BaseModel):
    project: str
    experiment_name: str
    entity: Optional[str] = None
    group: Optional[str] = None
    id: Optional[str] = None
    offline: bool = False


class LoggingConfig(BaseModel):
    log_lr: bool = True
    checkpoint_config: CheckpointConfig | None = None
    wandb_config: WandbConfig | None = None


class DataModuleArgs(BaseModel):
    """Int arguments for openfold3.core.data.framework.data_module"""

    datasets: list[TrainingDatasetSpec | InferenceDatasetSpec]
    batch_size: int = 1
    data_seed: int = 1234
    num_workers: int = 10
    epoch_len: int = 4
    num_epochs: int = 1000

    @classmethod
    def from_dataset_paths_and_configs(
        cls, dataset_specs: dict, dataset_paths: dict, **overrides
    ) -> list[TrainingDatasetSpec]:
        """Merge the dataset paths with the dataset specs."""
        values = overrides.copy()
        configs = []

        for mode, ds_specs in dataset_specs.items():
            for name, spec in ds_specs.items():
                spec["name"] = name
                spec["mode"] = mode
                spec["config"]["dataset_paths"] = dataset_paths[name]

                configs.append(TrainingDatasetSpec.model_validate(spec))

        values["datasets"] = configs
        return cls.model_validate(values)

    def to_data_module_config(self):
        return DataModuleConfig(
            datasets=self.datasets,
            batch_size=self.batch_size,
            data_seed=self.data_seed,
            num_workers=self.num_workers,
            epoch_len=self.epoch_len,
            num_epochs=self.num_epochs,
        )


class PlTrainerArgs(BaseModel):
    """Arguments for openfold3.core.pl_trainer.Trainer"""

    max_epochs: int = 1000  # pl_trainer default
    accelerator: str = "gpu"
    precision: int = 16
    num_nodes: int = 1
    profiler: Optional[str] = None
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True
    enable_model_summary: bool = False


class ExperimentConfig(BaseModel):
    mode: str
    output_dir: str
    mpi_plugin: bool = False
    deepspeed_config_path: str | None = None
    compile: bool = False
    num_gpus: int = 1

    pl_trainer_args: PlTrainerArgs
    model_update: ModelUpdate


class TrainableExperimentConfig(ExperimentConfig):
    seed: int = 42
    data_seed: int = 1234
    num_workers: int = 0
    epoch_len: int = 2
    restart_checkpoint_path: FilePathOrNone = None 

    logging_config: LoggingConfig | None = None
    dataset_paths: dict[str, TrainingDatasetPaths]
    model_update: ModelUpdate = ModelUpdate(presets=["train"])
    dataset_configs: dict[str, Any]
