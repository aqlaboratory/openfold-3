"""Treadmill logging utilities."""

import contextvars
import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from memory_profiler import profile

from openfold3.core.data.primitives.structure.interface import (
    get_query_interface_atom_pair_idxs,
)


@dataclass(frozen=False)
class ComplianceLog:
    """Dataclass to store compliance logs.

    Attributes:
        passed_ids:
            Set of PDB IDs that passed all asserts and didn't raise an error in the
            __getitem__.
    """

    passed_ids: set[str] = field(default_factory=set)

    def save_worker_compliance_file(self, worker_compliance_file: Path):
        """Saves the compliance log to a file."""
        passed_ids_list = list(self.passed_ids)
        pd.DataFrame(
            {
                "passed_ids": passed_ids_list,
            }
        ).to_csv(
            worker_compliance_file,
            mode="w",
            index=False,
            header=False,
            sep="\t",
        )

    @staticmethod
    def parse_compliance_file(compliance_file: Path):
        """Parses the compliance file and returns the instantiated class."""
        df = pd.read_csv(compliance_file, sep="\t", header=None)
        return ComplianceLog(
            passed_ids=set(df[0].tolist()),
        )


# Runtime context variables
def _default_factory_list():
    return []


def _default_factory_dict():
    return {}


LOG_RUNTIMES = contextvars.ContextVar("LOG_RUNTIME", default=False)
RUNTIME_DICT = contextvars.ContextVar("RUNTIME_DICT", default=_default_factory_dict())  # noqa: B039
# Memory context variables
LOG_MEMORY = contextvars.ContextVar("LOG_MEMORY", default=False)
WORKER_MEM_LOG_PATH = contextvars.ContextVar("WORKER_MEM_LOG_PATH", default=None)
MEM_PROFILED_FUNC_KEYS = contextvars.ContextVar(
    "MEM_PROFILED_FUNC_KEYS",
    default=_default_factory_list(),  # noqa: B039
)
# Template preprocessing context variables
TEMPLATE_PROCESS_LOGGER = contextvars.ContextVar(
    "TEMPLATE_PROCESS_LOGGER", default=None
)
# Mapping of function names to their respective runtime logging functions
F_NAME_ORDER = [
    "runtime-target-structure-proc",
    "runtime-target-structure-feat",
    "runtime-msa-proc",
    "runtime-msa-feat",
    "runtime-template-proc",
    "runtime-template-feat",
    "runtime-ref-conf-proc",
    "runtime-ref-conf-feat",
    "runtime-target-structure-proc-parse",
    "runtime-target-structure-proc-token",
    "runtime-target-structure-proc-crop",
    "runtime-target-structure-proc-expand",
    "runtime-msa-proc-crop-to-seq",
    #    "runtime-msa-proc-proc", # not logged
    "runtime-msa-proc-parse",
    "runtime-msa-proc-create-query",
    "runtime-msa-proc-homo-mono",
    "runtime-msa-proc-create-paired",
    "runtime-msa-proc-expand-paired",
    "runtime-msa-proc-create-main",
    "runtime-msa-proc-apply-crop",
]


def log_runtime_memory(
    runtime_enabled: bool = LOG_RUNTIMES,
    mem_enabled: bool = LOG_MEMORY,
    mem_stream: Path = WORKER_MEM_LOG_PATH,
    runtime_dict_key: str | None = None,
) -> callable:
    """Decorator factory to log the runtime or memory use of a function.

    Memory profiling is only ran for functions whose provided runtime_dict_key
    is listed in the MEM_PROFILED_FUNC_KEYS context variable. When no function
    names are provided for memory profiling and it is turned on, all functions
    in F_NAME_ORDER are profiled.

    Args:
        runtime_enabled (bool, optional):
            Whether to profile runtimes. Defaults to LOG_RUNTIMES context variable.
        mem_enabled (bool, optional):
            Whether to profile memory. Defaults to LOG_MEMORY context variable.
        mem_stream (Path, optional):
            Path to a log file into which the memory profiling log should be streamed.
            Defaults to WORKER_MEM_LOG_PATH context variable.
        runtime_dict_key (str | None, optional):
            String to use as key in the runtime dict collecting the runtimes in a
            hierarchy of decorated functions.

    Raises:
        RuntimeError:
            If both runtime and memory profiling are enabled.

    Returns:
        callable:
            Decorator function.
    """

    def decorator(func):
        """Actual runtime/memory profiler decorator."""

        function_name = (
            runtime_dict_key
            if runtime_dict_key is not None
            else f"{func.__module__}.{func.__qualname__}"
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function to allow for conditional profiling."""

            profile_runtime = runtime_enabled.get()
            profile_memory = mem_enabled.get()

            if profile_runtime & profile_memory:
                raise RuntimeError(
                    "The log_runtime_memory decorator can only be use with "
                    "EITHER runtime OR memory profiling but both were enabled."
                )

            # Runtime profiling
            if profile_runtime:
                # Fetch the shared runtime dictionary
                runtimes = RUNTIME_DICT.get()
                # The first time a decorated function is called within a context where
                # LOG_RUNTIMES is True, it initializes the RUNTIME_DICT
                if runtimes is None:
                    runtimes = {}
                    runtimes_token = RUNTIME_DICT.set(runtimes)
                # Subsequent decorated functions in the same context will use the
                # existing RUNTIME_DICT
                else:
                    runtimes_token = None

                # Measure the runtime
                start_time = time.time()
                result = func(*args, **kwargs)
                runtime = time.time() - start_time

                # Log the runtime to the context variable and the wrapper attribute
                runtimes[function_name] = runtime
                wrapper.runtime = runtimes

                # Reset the context variable to the original value
                if runtimes_token is not None:
                    RUNTIME_DICT.reset(runtimes_token)

            # Memory profiling
            elif profile_memory & (function_name in MEM_PROFILED_FUNC_KEYS.get()):
                # Use the context handler to set the memory log file path
                with open(mem_stream.get(), "a") as fp:
                    # Decorate the function with the memory profiler
                    profiled_func = profile(stream=fp)(func)
                    # Execute the function to profile memory
                    result = profiled_func(*args, **kwargs)
                wrapper.runtime = None

            # No profiling
            else:
                result = func(*args, **kwargs)
                wrapper.runtime = None
            return result

        wrapper.runtime = None
        return wrapper

    return decorator


def parse_memory_profiler_log(log_file_path):
    data = []
    code_block_data = []
    current_function = None
    current_pdb_id = None
    current_preferred_chain_or_interface = None

    # Regular expressions to match lines
    func_def_regex = re.compile(r"^\s*(\d+)\s+.*def\s+([^\s\(]+)")
    mem_line_regex = re.compile(
        r"^\s*(\d+)\s+([\d\.]+)\s+MiB\s+([\d\.\-\+]+)\s+MiB\s+(\d+)\s+(.*)"
    )
    pdb_id_regex = re.compile(r"^pdb_id:\s*(\S+)")
    chain_regex = re.compile(r"^preferred_chain_or_interface:\s*(\S+)")

    with open(log_file_path) as f:
        for line in f:
            line = line.rstrip("\n")
            # Skip empty lines and headers
            if not line.strip() or line.startswith(("Line #", "====")):
                continue

            # Check for pdb_id
            pdb_match = pdb_id_regex.match(line)
            if pdb_match:
                current_pdb_id = pdb_match.group(1)
                continue

            # Check for preferred_chain_or_interface
            chain_match = chain_regex.match(line)
            if chain_match:
                current_preferred_chain_or_interface = chain_match.group(1)
                # Now that we have both, process code_block_data
                if code_block_data:
                    for item in code_block_data:
                        item["pdb_id"] = current_pdb_id
                        item["preferred_chain_or_interface"] = (
                            current_preferred_chain_or_interface
                        )
                    data.extend(code_block_data)
                    code_block_data = []
                else:
                    # No data collected before pdb_id and preferred_chain_or_interface
                    pass
                # Reset pdb_id and chain
                current_pdb_id = None
                current_preferred_chain_or_interface = None
                continue

            # Check for function definition
            func_match = func_def_regex.match(line)
            if func_match:
                current_function = func_match.group(2)
                continue

            # Check for memory data lines
            mem_match = mem_line_regex.match(line)
            if mem_match:
                line_number = int(mem_match.group(1))
                mem_usage_mib = float(mem_match.group(2))
                increment_mib = float(mem_match.group(3))
                occurrences = int(mem_match.group(4))
                code_line = mem_match.group(5).strip()
                # # Skip lines that contain the @profile decorator
                # if '@profile' in code_line or '@log_runtime_memory' in code_line:
                #     continue
                # Convert MiB to MB
                mem_usage_mb = mem_usage_mib * 1.048576
                increment_mb = increment_mib * 1.048576
                code_block_data.append(
                    {
                        "function": current_function,
                        "line_number": line_number,
                        "mem_usage_MB": mem_usage_mb,
                        "increment_MB": increment_mb,
                        "occurrences": occurrences,
                        "code_line": code_line,
                    }
                )
    # At the end, if there is any remaining code_block_data
    if code_block_data:
        # We may not have pdb_id and preferred_chain_or_interface
        # For consistency, we can set them to None or empty string
        for item in code_block_data:
            item["pdb_id"] = current_pdb_id
            item["preferred_chain_or_interface"] = current_preferred_chain_or_interface
        data.extend(code_block_data)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Reorder columns
    df = df[
        [
            "pdb_id",
            "preferred_chain_or_interface",
            "function",
            "line_number",
            "mem_usage_MB",
            "increment_MB",
            "occurrences",
            "code_line",
        ]
    ]
    return df


def compute_interface(
    query_atom_array: AtomArray,
    target_atom_array: AtomArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the atom/chain pairs that form the interface between two structures.

    Optionally returns the residue pairs.

    Args:
        query_atom_array (AtomArray):
            Query atom array.
        target_atom_array (AtomArray):
            Target atom array.
        return_res_pairs (bool, optional):
            Whether to return an array of residue pairs. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            tuple of residue pairs and chain pairs
    """
    atom_pairs, chain_pairs = get_query_interface_atom_pair_idxs(
        query_atom_array,
        target_atom_array,
        distance_threshold=5.0,
        return_chain_pairs=True,
    )
    res_pairs = np.concatenate(
        [
            query_atom_array.res_id[atom_pairs[:, 0]][..., np.newaxis],
            target_atom_array.res_id[atom_pairs[:, 1]][..., np.newaxis],
        ],
        axis=1,
    )
    return res_pairs, chain_pairs


def encode_interface(res_pairs: np.ndarray, chain_pairs: np.ndarray) -> str:
    """Encodes the interface as a string.

    Args:
        res_pairs (np.ndarray):
            Array of residue index pairs.
        chain_pairs (np.ndarray):
            Array of chain index pairs.

    Returns:
        str:
            Encoded interface string. Has the format:
            "chain_id.res_id-chain_id.res_id;..."
    """
    unique_contacts = np.unique(
        np.concatenate([res_pairs, chain_pairs], axis=-1), axis=0
    )
    return ";".join(
        np.core.defchararray.add(
            np.core.defchararray.add(
                np.core.defchararray.add(
                    np.core.defchararray.add(unique_contacts[:, 2], "."),
                    unique_contacts[:, 0],
                ),
                "-",
            ),
            np.core.defchararray.add(
                np.core.defchararray.add(unique_contacts[:, 3], "."),
                unique_contacts[:, 1],
            ),
        )
    )


def get_interface_string(
    query_atom_array: AtomArray, target_atom_array: AtomArray
) -> str:
    """Computes the interface string between two structures.

    Args:
        query_atom_array (AtomArray):
            Query atom array.
        target_atom_array (AtomArray):
            Target atom array.
    Returns:
        str:
            Encoded interface string
    """
    r, c = compute_interface(
        query_atom_array,
        target_atom_array,
    )
    return encode_interface(r, c)


def decode_interface(interface_string: str) -> tuple[np.ndarray, np.ndarray]:
    """Decodes the interface string.

    Args:
        interface (str):
            Encoded interface string. Has the format:
            "chain_id.res_id-chain_id.res_id;..."

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Array of residue index pairs and chain index pairs.
    """
    contacts = interface_string.split(";")
    contacts = np.array([c.split("-") for c in contacts])
    contacts = np.char.split(contacts, ".")
    contacts_flattened = np.concatenate(contacts.ravel())

    chains = contacts_flattened[::2]
    residues = np.array(contacts_flattened[1::2], dtype=int).reshape(-1, 2)

    chain_residues = np.column_stack((residues, chains.reshape(-1, 2)))

    return chain_residues[:, :2], chain_residues[:, 2:]


def configure_template_logger(
    log_level: str, log_to_file: bool, log_to_console: bool, log_dir: Path
) -> logging.Logger:
    logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper())
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    if log_to_file:
        pid = os.getpid()
        file_handler = logging.FileHandler(log_dir / Path(f"process_{pid}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
