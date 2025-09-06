# args TODO add license

import json
import logging
import os
import sys
from pathlib import Path

import click
import pytorch_lightning as pl
import torch
import wandb
from deepspeed.utils import zero_to_fp32
from ml_collections import ConfigDict
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import DataModule
from openfold3.core.utils.precision_utils import OF3DeepSpeedPrecision
from openfold3.core.utils.script_utils import set_ulimits
from openfold3.projects import registry
from openfold3.projects.of3_all_atom.config.runner_file_checks import (
    _check_data_module_config,
)

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


def _configure_wandb_logger(
    wandb_args: ConfigDict, is_mpi_rank_zero: bool, output_dir: str
) -> WandbLogger:
    """Configures wandb and wandb logger."""

    wandb_id = wandb_args.id if hasattr(wandb_args, "id") else None

    wandb_init_dict = dict(
        project=wandb_args.project,
        entity=wandb_args.entity,
        group=wandb_args.group,
        name=wandb_args.experiment_name,
        dir=output_dir,
        resume="allow",
        reinit=True,
        id=wandb_id,
    )

    # Only initialize wandb for rank zero worker (MPI env), or else
    # each worker will generate a different id
    if is_mpi_rank_zero:
        wandb.run = wandb.init(**wandb_init_dict)

    run_offline = wandb_args.get("offline", False)
    wandb_logger = WandbLogger(
        **wandb_init_dict,
        save_dir=output_dir,
        log_model=False,
        offline=run_offline,
    )
    return wandb_logger


def get_model_state_dict_from_ds_checkpoint(checkpoint_dir):
    latest_path = os.path.join(checkpoint_dir, "latest")
    if os.path.isfile(latest_path):
        with open(latest_path) as fd:
            tag = fd.read().strip()
    else:
        raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)
    _DS_CHECKPOINT_VERSION = 2  # based on manual parsing of checkpoint files
    state_file = zero_to_fp32.get_model_state_file(
        ds_checkpoint_dir, _DS_CHECKPOINT_VERSION
    )
    return torch.load(state_file)


def restore_lr_step(ckpt_path: Path, lightning_module: pl.LightningModule):
    if ckpt_path.is_dir():
        sd = get_model_state_dict_from_ds_checkpoint(str(ckpt_path))
    else:
        sd = torch.load(str(ckpt_path))
    last_global_step = int(sd["global_step"])

    logger.info(f"Restoring last lr step {last_global_step} from {ckpt_path}")
    lightning_module.resume_last_lr_step(last_global_step)


@click.command()
@click.option(
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Yaml that specifies model and dataset parameters, see examples/runner.yml",
)
@click.option("--seed", type=int, help="Initial seed for all processes")
@click.option(
    "--data_seed",
    type=int,
    help="Initial seed for data pipeline. Defaults to seed if not specified.",
)
def main(runner_yaml: Path, seed: int, data_seed: int):
    runner_args = ConfigDict(config_utils.load_yaml(runner_yaml))

    # Set resource limits
    set_ulimits()

    # If specified, add seeds to runner dict to save to wandb
    if seed is not None:
        runner_args["seed"] = seed

    if data_seed is not None:
        runner_args["data_seed"] = data_seed

    if runner_args.get("log_level"):
        log_level = runner_args.get("log_level").upper()

        output_dir = Path(runner_args.get("output_dir"))
        output_dir.mkdir(exist_ok=True)
        log_filepath = output_dir / "console_logs.log"
        logging.basicConfig(filename=log_filepath, level=log_level, filemode="a")

    world_size = runner_args.num_gpus * runner_args.pl_trainer.num_nodes
    is_distributed = world_size > 1

    # Set seed
    seed = runner_args.get("seed")
    if seed is None and is_distributed:
        raise ValueError("For distributed training, seed must be specified")

    logger.info(f"Running with seed: {seed}")
    pl.seed_everything(seed, workers=True)

    project_entry = registry.get_project_entry(runner_args.project_type)

    project_config = registry.make_config_with_presets(
        project_entry, runner_args.presets
    )
    if runner_args.get("config_update"):
        project_config.update(runner_args.config_update)

    model_config = project_config.model
    lightning_module = project_entry.model_runner(
        model_config, _compile=runner_args.compile
    )

    dataset_config_builder = project_entry.dataset_config_builder
    data_module_config = registry.make_dataset_module_config(
        runner_args,
        dataset_config_builder,
        project_config,
    )
    _check_data_module_config(data_module_config)
    lightning_data_module = DataModule(data_module_config, world_size=world_size)

    loggers = []

    is_mpi = runner_args.get("mpi_plugin")
    cluster_environment = MPIEnvironment() if is_mpi else None

    # Select optimization strategy
    if runner_args.get("deepspeed_config_path"):
        strategy = DeepSpeedStrategy(
            config=runner_args.deepspeed_config_path,
            cluster_environment=cluster_environment,
            precision_plugin=OF3DeepSpeedPrecision(
                precision=runner_args.pl_trainer.precision
            ),
        )
        if not model_config.settings.optimizer.use_deepspeed_adam:
            strategy.config["zero_force_ds_cpu_optimizer"] = False
    elif is_distributed:
        strategy = DDPStrategy(
            find_unused_parameters=False, cluster_environment=cluster_environment
        )
    else:
        strategy = None

    is_mpi_rank_zero = is_mpi and cluster_environment.global_rank() == 0
    wandb_logger = None
    if runner_args.get("wandb") and (not is_mpi or is_mpi_rank_zero):
        wandb_logger = _configure_wandb_logger(
            runner_args.wandb, is_mpi_rank_zero, runner_args.output_dir
        )
        loggers.append(wandb_logger)

    # Set up trainer arguments and callbacks
    callbacks = []

    if runner_args.get("checkpoint"):
        callbacks.append(ModelCheckpoint(**runner_args.checkpoint.to_dict()))

    if runner_args.get("log_lr") and wandb_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer_args = runner_args.pl_trainer.to_dict()
    trainer_args.update(
        {
            "default_root_dir": runner_args.output_dir,
            "strategy": strategy,
            "callbacks": callbacks,
            "logger": loggers,
            "devices": runner_args.num_gpus,
            # If DeepSpeed is enabled, these values will be passed to the DS config
            "gradient_clip_val": model_config.settings.gradient_clipping,
            "gradient_clip_algorithm": "norm",
        }
    )

    trainer = pl.Trainer(**trainer_args)

    # TODO: Resuming lr scheduler is handled internally in the checkpoint in the
    #  "lr_schedulers" dict. Handle case where we resume from a different schedule
    ckpt_path = runner_args.get("restart_checkpoint_path")
    weights_only_path = runner_args.get("weights_only_checkpoint_path")

    run_exists = False
    if runner_args.get("wandb") and runner_args.wandb.get("id"):
        wandb_ckpt_dir = (
            Path(runner_args.output_dir) / "openfold3" / runner_args.wandb.id
        )
        run_exists = wandb_ckpt_dir.is_dir() and any(wandb_ckpt_dir.iterdir())

    if weights_only_path and not run_exists:
        ckpt_path = None

        ckpt_dict = torch.load(weights_only_path)

        logger.info(f"Restoring weights from {weights_only_path}")

        init_model_from_ema_weights = runner_args.get("init_model_from_ema_weights")
        if init_model_from_ema_weights:
            logger.info("Loading model from ema weights")
            lightning_module.load_state_dict(ckpt_dict["ema"]["params"], strict=False)
            # These won't get updated if "submodule_enabled_subset" in the ema config
            # is set and does not include these pretrained layers. This is just so the
            # final ema weights are in one dict
            lightning_module.ema.load_state_dict(ckpt_dict["ema"])
        else:
            lightning_module.load_state_dict(ckpt_dict["state_dict"], strict=False)
            lightning_module.ema.load_state_dict(ckpt_dict["ema"])

        reset_scheduler = runner_args.get("reset_scheduler")
        if not reset_scheduler:
            restore_lr_step(
                ckpt_path=Path(weights_only_path), lightning_module=lightning_module
            )
        else:
            logger.info("Resetting LR scheduler")

        logger.info(f"Restoring datamodule state from {weights_only_path}")
        lightning_data_module.load_state_dict(ckpt_dict["DataModule"])

        logger.info("Restoring fit loop counters")
        trainer.fit_loop.load_state_dict(ckpt_dict["loops"]["fit_loop"])

    # Determine if running on rank zero process
    if wandb_logger is not None and trainer.global_rank == 0:
        if runner_args.get("log_grads"):
            wandb_logger.watch(lightning_module, log="all", log_graph=False)

        wandb_experiment = wandb_logger.experiment

        # Save pip environment to wandb
        freeze_path = os.path.join(wandb_experiment.dir, "package_versions.txt")
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wandb_experiment.save(f"{freeze_path}")

        runner_yaml_path = os.path.join(wandb_experiment.dir, "runner.json")
        with open(runner_yaml_path, "w") as fp:
            json.dump(runner_args.to_dict(), fp, indent=4)
        wandb_experiment.save(runner_yaml_path)

        # Save data module config
        data_config_path = os.path.join(wandb_experiment.dir, "data_config.json")
        with open(data_config_path, "w") as fp:
            json.dump(data_module_config.to_dict(), fp, indent=4)
        wandb_experiment.save(data_config_path)

        model_config_path = os.path.join(wandb_experiment.dir, "model_config.json")
        with open(model_config_path, "w") as fp:
            json.dump(model_config.to_dict(), fp, indent=4)
        wandb_experiment.save(model_config_path)

        if runner_args.get("deepspeed_config_path"):
            wandb_experiment.save(runner_args.deepspeed_config_path)

    # Run process appropriate process
    logger.info(f"Running {runner_args.mode} mode.")
    # Training + validation / profiling
    if (runner_args.mode == "train") | (runner_args.mode == "profile"):
        if runner_args.mode == "profile":  # TODO Implement profiling
            raise NotImplementedError("Profiling mode not yet implemented.")
        else:
            trainer.fit(
                model=lightning_module,
                datamodule=lightning_data_module,
                ckpt_path=ckpt_path,
            )

    # Validation
    elif runner_args.mode == "eval":
        trainer.validate(
            model=lightning_module,
            datamodule=lightning_data_module,
            ckpt_path=ckpt_path,
        )

    # Testing
    elif runner_args.mode == "test":
        trainer.test(
            model=lightning_module,
            datamodule=lightning_data_module,
            ckpt_path=ckpt_path,
        )

    # Prediction == inference
    elif runner_args.mode == "predict":
        trainer.predict(
            model=lightning_module,
            datamodule=lightning_data_module,
            ckpt_path=ckpt_path,
        )
    else:
        raise ValueError(
            f"""Invalid mode argument: {runner_args.mode}. Choose one of "
            "'train', 'test', 'predict', 'profile'."""
        )


if __name__ == "__main__":
    main()
