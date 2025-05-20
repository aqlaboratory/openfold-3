from typing import Any, Optional

from pydantic import BaseModel


class CheckpointConfig(BaseModel):
    every_n_epochs: int
    auto_insert_metric_name: bool
    save_last: bool
    save_top_k: int


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str] = None
    group: Optional[str] = None
    id: Optional[str] = None
    experiment_name: str
    offline: bool = False


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

class TrainableExperimentConfig(BaseModel):
    mode: str
    project_type: str
    presets: list[str]
    mpi_plugin: bool
    seed: int
    batch_size: int
    num_gpus: int
    output_dir: str
    log_lr: bool
    compile: bool
    num_workers: int
    epoch_len: int
    restart_checkpoint_path: str
    deepspeed_config_path: str | None = None
    checkpoint: CheckpointConfig | None = None
    wandb: WandbConfig | None = None

    pl_trainer: dict[str, Any]
    dataset_paths: dict[str, Any]
    config_update: dict[str, Any]
    dataset_configs: dict[str, Any]
