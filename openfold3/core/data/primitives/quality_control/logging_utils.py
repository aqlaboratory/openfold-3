"""Treadmill logging utilities."""

import contextvars
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArray

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


# Specify the context variable for logging runtimes
# This should be imported and set in the dataset class' module
LOG_RUNTIMES = contextvars.ContextVar("LOG_RUNTIME", default=False)


def log_runtime(enabled=LOG_RUNTIMES):
    """Decorator factory to log the runtime of a function."""

    def decorator(func):
        """Actual runtime logging decorator."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function to allow for conditionally logging the runtime."""
            # Here, fetch the context variable to determine if logging is enabled
            if enabled.get():
                start_time = time.time()
                result = func(*args, **kwargs)
                runtime = time.time() - start_time
                wrapper.runtime = runtime
            else:
                result = func(*args, **kwargs)
                wrapper.runtime = None
            return result

        wrapper.runtime = None
        return wrapper

    return decorator


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


@staticmethod
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
