from abc import ABC, abstractmethod
from functools import cached_property
import logging

import pytorch_lightning as pl
import ml_collections as mlc
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from openfold3.core.data.framework.data_module import DataModule, DataModuleConfig
from openfold3.core.utils.precision_utils import OF3DeepSpeedPrecision
from openfold3.entry_points.validator import (
    DataModuleArgs,
    ExperimentRunnerSettings,
)
from openfold3.projects.af3_all_atom.config.dataset_configs import (
    InferenceConfig,
    InferenceDatasetSpec,
)
from openfold3.projects.af3_all_atom.project_entry import AF3ProjectEntry


class ExperimentRunner(ABC):
    """Abstract class for experiments"""

    def __init__(self, experiment_config: ExperimentRunnerSettings):
        self.mode = experiment_config.mode
        self.output_dir = experiment_config.output_dir
        self.deepspeed_config_path = experiment_config.deepspeed_config_path
        # this section should include arguments for strategy
        self.pl_trainer_args = experiment_config.pl_trainer_args
        self.mpi_plugin = experiment_config.mpi_plugin
        self.compile = experiment_config.compile
        self.num_gpus = experiment_config.num_gpus

        # typical model update config
        self.model_update = experiment_config.model_update

    @abstractmethod
    def setup(self) -> None:
        """Set up the experiment environment.

        This includes configuring logging, setting the random seed,
        and initializing WandB if enabled.
        """
        pass

    ###############
    # Model and dataset setup
    ###############
    @property
    def project_entry(self) -> AF3ProjectEntry:
        """Get the project entry from the registry."""
        return AF3ProjectEntry()

    @property
    def model_config(self) -> mlc.ConfigDict:
        """Retrieve the model configuration."""
        return self.project_entry.get_model_config_with_update(self.model_update)

    @cached_property
    def lightning_module(self) -> pl.LightningModule:
        """Instantiate and return the model."""
        return self.project_entry.model_runner(self.model_config, _compile=self.compile)

    @cached_property
    def ckpt_path(self) -> str | None:
        """Get the checkpoint path for the model."""
        pass

    @property
    @abstractmethod
    def data_module_args(self) -> DataModuleConfig:
        """Construct arguments for the data_module."""
        pass

    @cached_property
    def lightning_data_module(self):
        return DataModule(
            DataModuleConfig.model_validate(self.data_module_args),
            world_size=self.world_size,
        )

    ###############
    # Distributed properties
    ###############
    @cached_property
    def num_nodes(self) -> int:
        """Retrieves the number of nodes available for training."""
        return self.pl_trainer_args.num_nodes

    @property
    def world_size(self) -> int:
        """Compute the world size based on GPUs and nodes."""
        return self.num_gpus * self.num_nodes

    @property
    def is_distributed(self) -> bool:
        """Check if the training is distributed using the world size."""
        return self.world_size > 1

    @property
    def is_mpi(self) -> bool:
        """Check if MPI plugin is enabled."""
        return self.mpi_plugin

    @property
    def is_mpi_rank_zero(self) -> bool:
        """Check if the current process is rank zero in an MPI environment."""
        return self.is_mpi and self.cluster_environment.global_rank() == 0

    @property
    def cluster_environment(self) -> MPIEnvironment | None:
        """Return the MPI cluster environment if enabled."""
        return MPIEnvironment() if self.is_mpi else None

    @cached_property
    def strategy(self) -> DDPStrategy | DeepSpeedStrategy | None:
        """Determine and return the training strategy."""
        if self.deepspeed_config_path is not None:
            _strategy = DeepSpeedStrategy(
                config=self.deepspeed_config_path,
                cluster_environment=self.cluster_environment,
                precision_plugin=OF3DeepSpeedPrecision(
                    precision=self.pl_trainer_args.precision
                ),
            )

            _use_deepspeed_adam = (
                self.model_config.settings.optimizer.use_deepspeed_adam
            )
            if not _use_deepspeed_adam:
                _strategy.config["zero_force_ds_cpu_optimizer"] = False

            return _strategy

    ###############
    # Logging and Callbacks
    ###############

    @cached_property
    def callbacks(self):
        """Set up and return the list of training callbacks."""
        _callbacks = []
        return _callbacks

    @cached_property
    def loggers(self):
        """Retrieve the list of loggers to be used in the experiment."""
        _loggers = []
        return _loggers

    ###############
    # pl.Trainer class and run command
    ###############

    @cached_property
    def trainer(self) -> pl.Trainer:
        """Create and return the trainer instance."""
        trainer_args = self.pl_trainer_args.model_dump()
        trainer_args.update(
            {
                "default_root_dir": self.output_dir,
                "strategy": self.strategy,
                "callbacks": self.callbacks,
                "logger": self.loggers,
                "devices": self.num_gpus,
                # If DeepSpeed is enabled, these values will be passed to the DS config
                "gradient_clip_val": self.model_config.settings.gradient_clipping,
                "gradient_clip_algorithm": "norm",
            }
        )

        return pl.Trainer(**trainer_args)

    def run(self):
        """Run the experiment in the specified mode.

        Depending on the mode (train, eval, test, predict), the corresponding
        PyTorch Lightning method is invoked.
        """
        # Run process appropriate process
        logging.info(f"Running {self.mode} mode.")
        # Training + validation
        if self.mode == "train":
            target_method = self.trainer.fit
        elif self.mode == "profile":
            raise NotImplementedError("Profiling mode not yet implemented.")
        elif self.mode == "eval":
            target_method = self.trainer.validate
        elif self.mode == "test":
            target_method = self.trainer.test
        elif self.mode == "predict":
            target_method = self.trainer.predict
        else:
            raise ValueError(
                f"""Invalid mode argument: {self.mode}. Choose one of "
                "'train', 'test', 'predict', 'profile'."""
            )

        target_method(
            model=self.lightning_module,
            datamodule=self.lightning_data_module,
            ckpt_path=self.ckpt_path,
        )


class TrainingExperimentRunner(ExperimentRunner):
    """Training experiment builder."""

    def __init__(self, experiment_config):
        super().__init__(experiment_config)

        # set up of data module args
        self.seed = experiment_config.seed
        self.data_seed = experiment_config.data_seed
        self.restart_checkpoint_path = experiment_config.restart_checkpoint_path
        self.dataset_paths = experiment_config.dataset_paths
        self.dataset_configs = experiment_config.dataset_configs
        self.wandb_config = experiment_config.wandb_config
        self.log_level = experiment_config.log_level

    def setup(self) -> None:
        """Set up the experiment environment.

        This includes configuring logging, setting the random seed,
        and initializing WandB if enabled.
        """
        self._setup_logger()
        self._set_random_seed()
        if self.use_wandb:
            self._wandb_setup()

    @cached_property
    def data_module_config(self) -> DataModuleConfig:
        return DataModuleArgs.from_dataset_paths_and_configs(
            self.dataset_configs,
            self.dataset_paths,
            **self.data_module_args,
        ).to_data_module_config()

    @property
    def use_wandb(self):
        """Determine if WandB should be used.

        Returns:
            True if WandB configuration is provided and either
            not using MPI or is the MPI rank zero.
        """
        return self.wandb_config and (not self.is_mpi or self.is_mpi_rank_zero)

    def _wandb_setup(self) -> None:
        """Initialize WandB logging and store configuration files."""
        self.wandb = WandbHandler(
            self.wandb_config,
            self.is_mpi_rank_zero,
            self.output_dir,
        )
        self.wandb.store_configs(
            self.runner_args,
            self.data_module_config,
            self.model_config,
        )

    @property
    def use_wandb(self):
        """Determine if WandB should be used.

        Returns:
            True if WandB configuration is provided and either
            not using MPI or is the MPI rank zero.
        """
        return self.wandb_config and (not self.is_mpi or self.is_mpi_rank_zero)

    def _setup_logger(self) -> None:
        """Configure the logging settings.

        Sets the log level and log file path based on runner arguments.
        """
        log_level = self.log_level
        if log_level is None:
            return

        VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if not isinstance(log_level, str) or log_level.upper() not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LOG_LEVELS}")

        log_level = log_level.upper()
        log_filepath = self.output_dir / "console_logs.log"
        logging.basicConfig(filename=log_filepath, level=log_level, filemode="w")

    def _set_random_seed(self) -> None:
        """Set the random seed for reproducibility."""

        seed = self.seed
        if seed is None and self.is_distributed:
            raise ValueError("For distributed training, seed must be specified")

        if not isinstance(seed, int):
            raise ValueError(
                f"seed={seed} must be an integer. Please provide a valid seed."
            )

        logging.info(f"Running with seed: {seed}")
        pl.seed_everything(seed, workers=True)

    @cached_property
    def loggers(self):
        """Retrieve the list of loggers to be used in the experiment."""
        _loggers = []
        if self.use_wandb:
            _loggers.append(self.wandb.logger)
        return _loggers

    @cached_property
    def callbacks(self):
        """Set up and return the list of training callbacks."""
        _callbacks = []

        _checkpoint = self.runner_args.get("checkpoint")
        if _checkpoint is not None:
            _callbacks.append(ModelCheckpoint(**_checkpoint.to_dict()))

        _log_lr = self.runner_args.get("log_lr")
        if _log_lr is not None and self.use_wandb:
            _callbacks.append(LearningRateMonitor(logging_interval="step"))

        return _callbacks


class InferenceExperimentRunner(ExperimentRunner):
    """Training experiment builder."""

    def __init__(self, experiment_config, inference_query_set):
        super().__init__(experiment_config)

        # set up of data module args
        self.inference_query_set = inference_query_set
        self.dataset_paths = experiment_config.dataset_paths
        self.dataset_configs = experiment_config.dataset_configs

        # model path
        self.checkpoint_path = self.inference_settings.checkpoint_path

        # do we include args for msa handling here? Should be processed separately

    @cached_property
    def callbacks(self):
        """Set up and return the list of training callbacks."""
        _callbacks = [OF3OutputWriter(self.output_dir)]
        return _callbacks


class WandbHandler:
    """Handles WandB logger initialization and configuration storage.

    This class is responsible for setting up the WandB logger and saving
    the experiment configurations to WandB.
    """

    def __init__(
        self,
        wandb_args: None | ConfigDict,
        is_mpi_rank_zero: bool,
        output_dir: Path,
    ):
        """Initialize the WandbHandler.

        Args:
            wandb_args: The WandB related configuration.
            is_mpi_rank_zero: True if the current process is rank zero in an MPI setup.
            output_dir: The directory to store WandB files.
        """
        self.wandb_args = wandb_args
        self.output_dir = output_dir
        self.is_mpi_rank_zero = is_mpi_rank_zero
        self._logger = None

    def _init_logger(self) -> None:
        """Initialize the wandb environment and create the WandbLogger."""
        if self.wandb_args is None:
            raise ValueError("wandb_args must be provided to use wandb logger")

        wandb_id = self.wandb_args.id if hasattr(self.wandb_args, "id") else None

        wandb_init_dict = dict(
            project=self.wandb_args.project,
            entity=self.wandb_args.entity,
            group=self.wandb_args.group,
            name=self.wandb_args.experiment_name,
            dir=self.output_dir,
            resume="allow",
            reinit=True,
            id=wandb_id,
        )

        # Only initialize wandb for rank zero worker (MPI env), or else
        # each worker will generate a different id
        if self.is_mpi_rank_zero:
            wandb.run = wandb.init(**wandb_init_dict)

        run_offline = self.wandb_args.get("offline", False)
        self._logger = WandbLogger(
            **wandb_init_dict,
            save_dir=self.output_dir,
            log_model=False,
            offline=run_offline,
        )

    @property
    def logger(self) -> WandbLogger:
        """Return the WandB logger instance. The logger is initialized
        on first access."""
        if self._logger is None:
            self._init_logger()
        assert self._logger is not None
        return self._logger

    def store_configs(
        self,
        runner_args: ConfigDict,
        data_module_config: DataModuleConfig,
        model_config: ConfigDict,
    ) -> None:
        """Store experiment configuration files to the WandB run directory.

        This method saves the pip freeze output, runner configuration,
        data module configuration, and model configuration as files in
        the WandB run.

        Args:
            runner_args: The runner configuration.
            data_module_config: The configuration for the data module.
            model_config: The configuration for the model.
        """

        wandb_experiment = self.logger.experiment
        # Save pip environment to wandb
        freeze_path = os.path.join(wandb_experiment.dir, "package_versions.txt")
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wandb_experiment.save(f"{freeze_path}")

        # user given runner yaml
        runner_yaml_path = os.path.join(wandb_experiment.dir, "runner.json")
        with open(runner_yaml_path, "w") as fp:
            json.dump(runner_args.to_dict(), fp, indent=4)
        wandb_experiment.save(runner_yaml_path)

        # save the deepspeed config if it exists
        if runner_args.get("deepspeed_config_path"):
            wandb_experiment.save(runner_args.deepspeed_config_path)

        # Save data module config
        data_config_path = os.path.join(wandb_experiment.dir, "data_config.json")
        with open(data_config_path, "w") as fp:
            fp.write(data_module_config.model_dump_json(indent=4))
        wandb_experiment.save(data_config_path)

        # Save model config
        model_config_path = os.path.join(wandb_experiment.dir, "model_config.json")
        with open(model_config_path, "w") as fp:
            json.dump(model_config.to_dict(), fp, indent=4)
        wandb_experiment.save(model_config_path)
