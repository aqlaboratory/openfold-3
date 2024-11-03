""" "Treadmill helper functions for configuring the worker init function."""

import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from openfold3.core.data.primitives.quality_control.logging_utils import (
    F_NAME_ORDER,
    LOG_MEMORY,
    LOG_RUNTIMES,
    MEM_PROFILED_FUNC_KEYS,
    WORKER_MEM_LOG_PATH,
    ComplianceLog,
)


def configure_context_variables(
    log_runtimes: bool,
    log_memory: bool,
    worker_dataset: Dataset,
    mem_profiled_func_keys: str | None,
) -> tuple[int, int, int, int]:
    """Configures the context variables for the worker.

    Also assigns the context variable state tokens to the worker-specific copy
    of the dataset.

    Args:
        log_runtimes (bool):
            Whether to log runtimes.
        log_memory (bool):
            Whether to log memory.
        worker_dataset (Dataset):
            Worker-specific copy of the dataset.
        mem_profiled_func_keys (str | None):
            List of function keys for which to profile memory.

    Returns:
        tuple[int, int, int, int]:
            Context variable state tokens for runtime logging, memory logging, and
            memory log path.
    """

    # Convert the comma separated string of function keys to a list
    if mem_profiled_func_keys is not None:
        mem_profiled_func_keys = [s.strip() for s in mem_profiled_func_keys.split(",")]

        if len(set(mem_profiled_func_keys) - set(F_NAME_ORDER)) != 0:
            raise RuntimeError(
                "Invalid function keys were provided for memory profiling. "
                f"The set of valid function keys: {F_NAME_ORDER}."
            )
    else:
        mem_profiled_func_keys = []

    # Set context variables
    runtime_token = LOG_RUNTIMES.set(log_runtimes)
    mem_token = LOG_MEMORY.set(log_memory)
    mem_log_token = WORKER_MEM_LOG_PATH.set(
        worker_dataset.get_worker_path(subdirs=None, fname="memory_profile.log")
    )
    mem_func_token = MEM_PROFILED_FUNC_KEYS.set(
        F_NAME_ORDER if (len(mem_profiled_func_keys) == 0) else mem_profiled_func_keys
    )
    # Assign context variable state tokens to the worker-specific copy of the dataset
    worker_dataset.runtime_token = runtime_token
    worker_dataset.mem_token = mem_token
    worker_dataset.mem_log_token = mem_log_token
    worker_dataset.mem_func_token = mem_func_token


def configure_worker_init_func_logger(
    worker_id: int, worker_dataset: Dataset, log_level: str, log_output_directory: Path
) -> logging.Logger:
    """Configures the logger for the worker.

    Also assigns the worker-specific logger to the worker-specific copy of the
    dataset.

    Args:
        worker_id (int):
            Worker id.
        worker_dataset (Dataset):
            Worker-specific copy of the dataset.
        log_level (str):
            Logging level.
        log_output_directory (Path):
            Treadmill output directory.

    Returns:
        logging.Logger:
            Worker logger.
    """
    # Configure logging
    worker_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level)
    worker_logger.setLevel(numeric_level)

    # Clear any existing handlers
    if worker_logger.hasHandlers():
        worker_logger.handlers.clear()

    # Create a handler for each worker and corresponding dir
    worker_dir = log_output_directory / Path(f"worker_{worker_id}")
    worker_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(worker_dir / Path(f"worker_{worker_id}.log"))
    formatter = logging.Formatter(
        "%(asctime)s - Worker %(worker_id)s - " "%(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    worker_logger.addHandler(handler)

    # Add worker_id and log_output_directory to the logger (for formatting)
    worker_logger = logging.LoggerAdapter(
        worker_logger,
        {"worker_id": worker_id, "log_output_directory": log_output_directory},
    )

    # Set the logger to the local copy of the dataset in the current worker
    worker_dataset.logger = worker_logger
    return worker_logger


def configure_extra_data_file(
    worker_id: int,
    worker_dataset: Dataset,
    save_statistics: bool,
    log_runtimes: bool,
    log_output_directory: Path,
) -> None:
    """Configures the extra data file for the worker.

    Args:
        worker_id (int):
            Worker identifier.
        worker_dataset (Dataset):
            Worker-specific copy of the dataset.
        save_statistics (bool):
            Whether to save statistics.
        log_runtimes (bool):
            Whether to log runtimes.
        log_output_directory (Path):
            Treadmill output directory.
    """
    if save_statistics:
        all_headers = [
            "pdb-id",
            "chain-or-interface",
            "atoms",
            "atoms-crop",
            "atoms-protein",
            "atoms-protein-crop",
            "atoms-rna",
            "atoms-rna-crop",
            "atoms-dna",
            "atoms-dna-crop",
            "atoms-ligand",
            "atoms-ligand-crop",
            "res-protein",
            "res-protein-crop",
            "res-rna",
            "res-rna-crop",
            "res-dna",
            "res-dna-crop",
            "atoms-unresolved",
            "res-unresolved",
            "atoms-unresolved-crop",
            "res-unresolved-crop",
            "chains",
            "chains-crop",
            "entities-protein",
            "entities-protein-crop",
            "entities-rna",
            "entities-rna-crop",
            "entities-dna",
            "entities-dna-crop",
            "entities-ligand",
            "entities-ligand-crop",
            "msa-depth",
            "msa-num-paired-seqs",
            "msa-aligned-cols",
            "templates",
            "templates-aligned-cols",
            "tokens",
            "tokens-crop",
            "res-special-protein",
            "res-covmod-protein",
            "res-special-protein-crop",
            "res-covmod-protein-crop",
            "res-special-rna",
            "res-covmod-rna",
            "res-special-rna-crop",
            "res-covmod-rna-crop",
            "res-special-dna",
            "res-covmod-dna",
            "res-special-dna-crop",
            "res-covmod-dna-crop",
            "gyration-radius",
            "gyration-radius-crop",
            "interface-protein-protein",
            "interface-protein-protein-crop",
            "interface-protein-rna",
            "interface-protein-rna-crop",
            "interface-protein-dna",
            "interface-protein-dna-crop",
            "interface-protein-ligand",
            "interface-protein-ligand-crop",
        ]

        if log_runtimes:
            all_headers += F_NAME_ORDER

        full_extra_data_file = log_output_directory / Path("datapoint_statistics.tsv")
        if full_extra_data_file.exists():
            worker_dataset.logger.info(
                "Parsing processed datapoints from " f"{full_extra_data_file}."
            )
            df = pd.read_csv(full_extra_data_file, sep="\t", na_values=["NaN"])
            worker_dataset.processed_datapoint_log = list(set(df["pdb-id"]))
        else:
            worker_dataset.processed_datapoint_log = []

        worker_extra_data_file = log_output_directory / Path(
            f"worker_{worker_id}/datapoint_statistics.tsv"
        )

        with open(worker_extra_data_file, "w") as f:
            f.write("\t".join(all_headers) + "\n")

    else:
        worker_dataset.processed_datapoint_log = []


def configure_compliance_log(
    worker_dataset: Dataset, log_output_directory: Path
) -> None:
    """Assigns a compliance log to the dataset of a given worker.

    Loads an existing compliance file into a compliance log object for the worker.

    Args:
        worker_dataset (Dataset):
            Worker-specific copy of the dataset.
        log_output_directory (Path):
            Treadmill output directory.
    """
    compliance_file_path = log_output_directory / Path("passed_ids.tsv")

    if compliance_file_path.exists():
        worker_dataset.compliance_log = ComplianceLog.parse_compliance_file(
            compliance_file_path
        )

    else:
        worker_dataset.compliance_log = ComplianceLog(
            passed_ids=set(),
        )
