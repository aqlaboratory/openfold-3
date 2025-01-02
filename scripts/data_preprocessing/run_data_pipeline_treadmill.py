"""
Script to iterate over datapoints with the data pipeline.

Components:
- WeightedPDBDatasetWithLogging:
    - Custom WeightedPDBDataset class that catches asserts and exceptions in the
    __getitem__.
    - also allows for saving features and atom array when an exception occurs
    in a worker process.
- worker_init_function_with_logging:
    Custom worker init function with per-worker logging and feature/atom array saving.

The treadmill requires at least one worker to run.

Ways to run the treadmill:
1. Quality control mode
    > runs asserts and optionally saves features, atom array and the full exception
    traceback when an exception occurs
    - run_asserts=True
    - save_features=on_error/False
    - save_atom_array=on_error/False
    - save_full_traceback=True/False
2. Cropped atom array/feature precomputation mode
    > saves features and atom array for each datapoint
    - run_asserts=True/False
    - save_features=per_datapoint
    - save_atom_array=per_datapoint
    - save_full_traceback=True/False
3. Statistics logging mode
    > saves additional data to save during data processing
    - save_statistics=True
4. Runtime logging mode
    > logs runtimes of subpipelines during data processing to worker log files
    - log_runtimes=True
5. Runtime logging mode with statistics logging
    > logs runtimes of subpipelines into a  statistics file alongside the other
    collected statistics
    - save_statistics=True
    - log_runtimes=True
6. Memory logging mode
    > logs memory use of subpipelines during data processing to memory log files
    - log_memory=True

Mutually exclusive options:
 - run_asserts v. save_statistics
 - log_runtimes v. log_memory
"""

import os
import random
import sys
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import _NUMPY_AVAILABLE
from openfold3.core.data.framework.lightning_utils import _generate_seed_sequence
from openfold3.core.data.framework.single_datasets.abstract_single import (
    DATASET_REGISTRY,
)
from openfold3.core.data.primitives.quality_control.logging_datasets import (
    add_logging_to_dataset,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    parse_memory_profiler_log,
)
from openfold3.core.data.primitives.quality_control.worker_config import (
    configure_compliance_log,
    configure_context_variables,
    configure_extra_data_file,
    configure_worker_init_func_logger,
)
from openfold3.projects import registry

np.set_printoptions(threshold=sys.maxsize)


@click.command()
@click.option(
    "--runner-yml-file",
    required=True,
    help="Yaml that specifies model and dataset parameters," "see examples/runner.yml",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--seed",
    required=True,
    help="Seed for reproducibility",
    type=int,
)
@click.option(
    "--with-model-fwd",
    default=False,
    help="Whether to run the model forward pass with the produced features",
    type=bool,
)
@click.option(
    "--log-output-directory",
    required=True,
    help="Path to directory where logs will be saved",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=True),
    help="Set the logging level",
)
@click.option(
    "--run-asserts",
    default=False,
    type=bool,
    help="Whether to run asserts. If True and there exists a passed_ids.tsv file in "
    "log-output-directory, the treadmill will skip all fully compliant datapoints."
    " Otherwise a new passed_ids.tsv file will be created.",
)
@click.option(
    "--save-features",
    default="False",
    type=click.Choice(["on_error", "per_datapoint", "False"]),
    help=(
        "Whether to save the FeatureDict.  If on_error, saves when an exception occurs,"
        "if per_datapoint, saves for each datapoint AND when an exception occurs"
    ),
)
@click.option(
    "--save-atom-array",
    default="False",
    type=click.Choice(["on_error", "per_datapoint", "False"]),
    help=(
        "Whether to save the cropped atom array. If on_error, saves when an exception "
        "occurs, if per_datapoint, saves for each datapoint AND when an exception "
        "occurs."
    ),
)
@click.option(
    "--save-full-traceback",
    default=False,
    type=bool,
    help="Whether to save the tracebacks upon assert-fail or exception.",
)
@click.option(
    "--save-statistics",
    type=bool,
    default=False,
    help="Whether to save additional data to save during data processing.",
)
@click.option(
    "--log-runtimes",
    type=bool,
    default=False,
    help=(
        "Whether to log runtimes of subpipelines during data processing. By default, "
        "runtimes are logged in the worker log files. If True and save_statistics "
        "is True, the runtime of each subpipeline will be logged in the "
        "datapoint_statistics.tsv file instead."
    ),
)
@click.option(
    "--log-memory",
    type=bool,
    default=False,
    help="Whether to log memory use of subpipelines during data processing.",
)
@click.option(
    "--mem-profiled-func-keys",
    default=None,
    help=(
        "String of comma-separated function keys for which to profile memory. If not "
        "provided with log-memory=True, all functions specified in "
        "logging_utils.F_NAME_ORDER will be profiled."
    ),
)
@click.option(
    "--subset-to-examples",
    default="",
    type=str,
    help=(
        "Comma separated list of PDB IDs use to subset the dataset cache to run "
        "asserts on."
    ),
)
@click.option(
    "--no-preferred-chain-or-interface",
    type=bool,
    default=False,
    help="Whether to log memory use of subpipelines during data processing.",
)
def main(
    runner_yml_file: Path,
    seed: int,
    with_model_fwd: bool,
    log_output_directory: Path,
    log_level: str,
    run_asserts: bool,
    save_features: bool,
    save_atom_array: bool,
    save_full_traceback: bool,
    save_statistics: bool,
    log_runtimes: bool,
    log_memory: bool,
    mem_profiled_func_keys: str | None,
    subset_to_examples: str,
    no_preferred_chain_or_interface: bool,
) -> None:
    """Main function for running the data pipeline treadmill.

    Args:
        runner_yml_file (Path):
            File path to the input yaml file.
        seed (int):
            Seed to use for data pipeline.
        with_model_fwd (bool):
            Whether to run the model forward pass with the produced features.
        log_level (str):
            Logging level.
        run_asserts (bool):
            Whether to run asserts. If True and there exists a passed_ids.tsv file in
            log-output-directory, the treadmill will skip all fully compliant
            datapoints. Otherwise a new passed_ids.tsv file will be created.
        save_features (bool):
            Whether to run asserts. If on_error, saves when an exception occurs, if
            per_datapoint, saves for each datapoint AND when an exception occurs
        save_atom_array (bool):
            Whether to save atom array when an exception occurs.
        save_full_traceback (bool):
            Whether to save the per-sample full traceback when an exception occurs.
        save_statistics (bool):
            Whether to save additional data to save during data processing. If True and
            there exists a datapoint_statistics.tsv file in log-output-directory, the
            treadmill will skip all datapoints whose statistics have already been
            logged. Otherwise a datapoint_statistics.tsv file will be created for each
            worker and then collated into a single datapoint_statistics.tsv file in
            log-output-directory.
        log_runtimes (bool):
            Whether to log runtimes of subpipelines during data processing. By default,
            runtimes are logged in the worker log files. If True and save_statistics is
            True, the runtime of each subpipeline will be logged in the
            datapoint_statistics.tsv file instead.
        log_memory (bool):
            Whether to log memory use of subpipelines during data processing. Memory use
            is logged per-line for each function in the getitem into a
            memory_profile.log file in log-output-directory.
        mem_profiled_func_keys (list[str] | None):
            List of function keys for which to profile memory.
        subset_to_examples: (str)
            Comma separated list of PDB IDs use to subset the dataset cache to run
            asserts on.
        no_preferred_chain_or_interface (bool):
            Whether to sample a crop without a preferred chain or interface. Also
            let the treadmill skip all variations of each sample with all preferred
            chains and interfaces.

    Raises:
        ValueError:
            If num_workers < 1 or if more than one of run_asserts, log_runtimes, and
            log_memory are set to True.
        NotImplementedError:
            If with_model_fwd is True.

    Returns:
        None
    """
    # Set seed
    pl.seed_everything(seed, workers=False)

    # Parse runner yml file and init Dataset
    runner_args = ConfigDict(config_utils.load_yaml(runner_yml_file))

    # Run checks on the input args
    run_arg_checks(
        runner_args,
        run_asserts,
        save_statistics,
        log_runtimes,
        log_memory,
    )

    project_entry = registry.get_project_entry(runner_args.project_type)
    project_config = registry.make_config_with_presets(
        project_entry, runner_args.presets
    )
    dataset_config_builder = project_entry.dataset_config_builder
    data_module_config = registry.make_dataset_module_config(
        runner_args,
        dataset_config_builder,
        project_config,
    )
    if len(data_module_config.datasets) > 1:
        raise NotImplementedError(
            "Running the treadmill script with multiple datasets is not yet "
            "implemented."
        )

    dataset_settings = data_module_config.datasets[0]
    LoggingDataset = add_logging_to_dataset(
        DATASET_REGISTRY[dataset_settings.get("class")]
    )
    # add_stochastic_sampler_to_dataset here?
    logging_dataset = LoggingDataset(
        run_asserts=run_asserts,
        save_features=save_features,
        save_atom_array=save_atom_array,
        save_full_traceback=save_full_traceback,
        save_statistics=save_statistics,
        log_runtimes=log_runtimes,
        log_memory=log_memory,
        subset_to_examples=subset_to_examples,
        no_preferred_chain_or_interface=no_preferred_chain_or_interface,
        dataset_config=dataset_settings.get("config"),
    )

    # This function needs to be defined here to form a closure
    # around log_output_directory, log_level and save_statistics
    def worker_init_function_with_logging(
        worker_id: int, rank: int | None = None
    ) -> None:
        """Modified default Lightning worker_init_fn with logging.

        This worker_init_fn enables decoupling stochastic processes in the data
        pipeline from those in the model. Taken from Pytorch Lightning 2.4.1 source
        code: https://github.com/Lightning-AI/pytorch-lightning/blob/f3f10d460338ca8b2901d5cd43456992131767ec/src/lightning/fabric/utilities/seed.py#L85

        Args:
            worker_id (int):
                Worker id.
            rank (Optional[int], optional):
                Worker process rank. Defaults to None.
        """
        # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
        global_rank = rank if rank is not None else rank_zero_only.rank
        process_seed = torch.initial_seed()
        # back out the base seed so we can use all the bits
        base_seed = process_seed - worker_id
        seed_sequence = _generate_seed_sequence(
            base_seed, worker_id, global_rank, count=4
        )
        torch.manual_seed(seed_sequence[0])  # torch takes a 64-bit seed
        random.seed(
            (seed_sequence[1] << 32) | seed_sequence[2]
        )  # combine two 64-bit seeds
        if _NUMPY_AVAILABLE:
            import numpy as np

            np.random.seed(
                seed_sequence[3] & 0xFFFFFFFF
            )  # numpy takes 32-bit seed only

        # Get worker dataset
        worker_info = get_worker_info()
        worker_dataset = worker_info.dataset

        # Configure logger and log process & worker IDs
        worker_logger = configure_worker_init_func_logger(
            worker_id, worker_dataset, log_level, log_output_directory
        )
        worker_logger.info("Worker init function completed.")
        worker_logger.info(
            "logger worker ID: {}".format(worker_logger.extra["worker_id"])
        )
        worker_logger.info(f"process ID: {os.getpid()}")

        # Configure data file
        configure_extra_data_file(
            worker_id,
            worker_dataset,
            save_statistics,
            log_runtimes,
            log_output_directory,
        )

        # Configure compliance file
        configure_compliance_log(worker_dataset, log_output_directory)

        # Configure context variables
        configure_context_variables(
            log_runtimes, log_memory, worker_dataset, mem_profiled_func_keys
        )

    # Configure DataLoader
    data_loader = DataLoader(
        dataset=logging_dataset,
        batch_size=data_module_config.batch_size,
        num_workers=data_module_config.num_workers,
        worker_init_fn=worker_init_function_with_logging,
    )

    # Init model
    if with_model_fwd:
        raise NotImplementedError(
            "Running the treadmill script with model forward pass"
            " is not yet implemented."
        )

    # Iterate over dataset - catch interruptions
    try:
        for _ in tqdm(
            data_loader,
            desc="Iterating over WeightedPDBDataset",
            total=len(logging_dataset),
        ):
            pass
    finally:
        # Collate passed IDs from all workers
        if run_asserts:
            all_passed_ids = set()
            for worker_id in range(runner_args.num_workers):
                worker_compliance_file = log_output_directory / Path(
                    f"worker_{worker_id}/passed_ids.tsv"
                )
                if worker_compliance_file.exists():
                    df_worker = pd.read_csv(
                        worker_compliance_file, sep="\t", header=None
                    )
                    passed_ids_worker = set(df_worker[0].tolist())
                    all_passed_ids.update(passed_ids_worker)
                    worker_compliance_file.unlink()

            pd.DataFrame({"passed_ids": list(all_passed_ids)}).to_csv(
                log_output_directory / Path("passed_ids.tsv"),
                sep="\t",
                header=False,
                index=False,
            )
        # Collate the extra data from different workers
        if save_statistics:
            df_all = pd.DataFrame()
            for worker_id in range(runner_args.num_workers):
                worker_extra_data_file = log_output_directory / Path(
                    f"worker_{worker_id}/datapoint_statistics.tsv"
                )
                if worker_extra_data_file.exists():
                    df_all = pd.concat(
                        [
                            df_all,
                            pd.read_csv(
                                worker_extra_data_file, sep="\t", na_values=["NaN"]
                            ),
                        ]
                    )
                    worker_extra_data_file.unlink()

            # Save to single file or append to existing file
            full_extra_data_file = log_output_directory / Path(
                "datapoint_statistics.tsv"
            )
            df_all.to_csv(
                full_extra_data_file,
                sep="\t",
                index=False,
                na_rep="NaN",
                header=not full_extra_data_file.exists(),
                mode="a",
            )
        # Collate memory logs
        if log_memory:
            # Convert memory profile logs to dataframes
            df_all = pd.DataFrame()
            for worker_id in range(runner_args.num_workers):
                worker_memory_file = log_output_directory / Path(
                    f"worker_{worker_id}/memory_profile.log"
                )
                if worker_memory_file.exists():
                    df_all = pd.concat(
                        [df_all, parse_memory_profiler_log(worker_memory_file)]
                    )
                    worker_memory_file.unlink()

            full_worker_memory_file = log_output_directory / Path("memory_profile.tsv")
            df_all.to_csv(
                full_worker_memory_file,
                sep="\t",
                index=False,
                header=not full_worker_memory_file.exists(),
                mode="a",
            )
        # Collate logs
        combined_log = log_output_directory / Path("worker_logs.log")
        with combined_log.open("w") as out_file:
            for worker_id in range(runner_args.num_workers):
                worker_dir = log_output_directory / Path(f"worker_{worker_id}")
                worker_log = worker_dir / Path(f"worker_{worker_id}.log")
                out_file.write(f"Log file: {worker_log.name}\n")
                out_file.write(worker_log.read_text())
                worker_log.unlink()
                if not any(worker_dir.iterdir()):
                    worker_dir.rmdir()


def run_arg_checks(
    runner_args: ConfigDict,
    run_asserts: bool,
    save_statistics: bool,
    log_runtimes: bool,
    log_memory: bool,
) -> None:
    if runner_args.num_workers < 1:
        raise ValueError("This script only works with num_workers >= 1.")
    if sum([run_asserts, save_statistics]) > 1:
        raise ValueError(
            "Only one of run_asserts and save_statistics can be set to True."
        )
    if sum([log_runtimes, log_memory]) > 1:
        raise ValueError("Only one of log_runtimes, and log_memory can be set to True.")
    if log_memory & (runner_args.num_workers > 1):
        warnings.warn(
            (
                "Memory logging with more than one worker (currently using "
                f"{runner_args.num_workers}) may significantly slow down the treadmill "
                "iteration time."
            ),
            stacklevel=2,
        )


if __name__ == "__main__":
    main()

# TODOs:
# 6. implement the model forward pass
# 8. Add logic to re-crop the structure if the number of tokens is larger than the
# token budget - the number of re-crops and featurizations should be determined
# dynamically and in a way that likely covers the entire structure but with a
# maximun number of re-crops
# 10. add weighted stochastic sampling
# 11. add support for multiple datasets
