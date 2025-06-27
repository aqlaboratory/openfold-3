import json
import pickle as pkl
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, Union

import lmdb
from tqdm import tqdm

if TYPE_CHECKING:
    from openfold3.core.data.primitives.caches.format import DatasetCache

K = TypeVar("K")
V = TypeVar("V")


def convert_datacache_to_lmdb(
    dataset_cache_file_or_obj: Union[Path, "DatasetCache"],
    lmdb_directory: Path,
    map_size: int,
    mode: Literal["single-read", "iterative"] = "single-read",
    str_encoding: Literal["utf-8", "pkl"] = "utf-8",
    structure_data_encoding: Literal["utf-8", "pkl"] = "pkl",
    reference_molecule_data_encoding: Literal["utf-8", "pkl"] = "pkl",
) -> None:
    """Convert a DataCache JSON file to an LMDB directory.

    Args:
        json_file (Path | DatasetCache):
            The datacache JSON file to convert or an existing DatasetCache object.
        lmdb_directory (Path):
            The LMDB dir to which the data and lock files are to be written.
        map_size (int):
            Size of the json file in bytes, for example  2 * (1024**3) for a 2GB file.
            Provide a value slightly larger than the actual size of the json file.
        mode (Literal["single-read", "iterative"]):
            The mode to use to parse the json file. Can be one of 'single-read' or
            'iterative'. Use 'single-read' for small files and 'iterative' for large
            files.
        str_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the cache keys and _type and name values.
        structure_data_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the structure_data values. The 'pkl' encoding saves
            the dataclasses directly, whereas 'utf-8' encoding requires re-creating the
            dataclasses.
        reference_molecule_data_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the reference_molecule_data values.The 'pkl'
            encoding saves the dataclasses directly, whereas 'utf-8' encoding requires
            re-creating the dataclasses.
    """
    from openfold3.core.data.io.dataset_cache import (
        convert_dataclass_to_dict,
        read_datacache,
    )

    if mode == "single-read":
        dataset_cache = read_datacache(dataset_cache_file_or_obj)

        lmdb_env = lmdb.open(str(lmdb_directory), map_size=map_size, subdir=True)

        with lmdb_env.begin(write=True) as transaction:
            print("1/4: Adding _type to the LMDB.")
            transaction.put(
                b"_type",
                json.dumps(dataset_cache._type).encode(str_encoding),
            )
            print("2/4: Adding name to the LMDB.")
            transaction.put(
                b"name",
                json.dumps(dataset_cache.name).encode(str_encoding),
            )

            # Store each entry in structure_data separately
            for sdata_key, sdata_value in tqdm(
                dataset_cache.structure_data.items(),
                desc="3/4: Adding structure_data to the LMDB",
                total=len(dataset_cache.structure_data),
            ):
                key_bytes = f"structure_data:{sdata_key}".encode(str_encoding)
                if structure_data_encoding == "pkl":
                    val_bytes = pkl.dumps(sdata_value)
                else:
                    sdata_value_dict = convert_dataclass_to_dict(sdata_value)
                    val_bytes = json.dumps(sdata_value_dict).encode(
                        structure_data_encoding
                    )
                transaction.put(key_bytes, val_bytes)

            # Store each entry in reference_molecule_data separately
            for ref_mol_key, ref_mol_info in tqdm(
                dataset_cache.reference_molecule_data.items(),
                desc="4/4: Adding reference_molecule_data to the LMDB",
                total=len(dataset_cache.reference_molecule_data),
            ):
                key_bytes = f"reference_molecule_data:{ref_mol_key}".encode(
                    str_encoding
                )
                if reference_molecule_data_encoding == "pkl":
                    val_bytes = pkl.dumps(ref_mol_info)
                else:
                    ref_mol_info_dict = convert_dataclass_to_dict(ref_mol_info)
                    val_bytes = json.dumps(ref_mol_info_dict).encode(
                        reference_molecule_data_encoding
                    )
                transaction.put(key_bytes, val_bytes)

        lmdb_env.close()

    elif mode == "iterative":
        # TODO add logic to iteratively read the cache with ijson and write to LMDB
        # should be useful for super large caches
        raise NotImplementedError("Iterative mode is not yet implemented.")


class LMDBDict(Mapping[K, V], Generic[K, V]):
    def __init__(
        self,
        lmdb_env: lmdb.Environment,
        prefix: str,
        separator: chr = ":",
        key_encoding: Literal["utf-8", "pkl"] = "utf-8",
        value_encoding: Literal["utf-8", "pkl"] = "pkl",
    ):
        """A dict-like class with an LMDB backend for lazy loading of datacache entries.

        Args:
            lmdb_env (lmdb.Environment):
                The LMDB environment object.
            prefix (str): header for fields used to construct keys in lmdb
            separator (chr): Single separator character used to construct key
            key_encoding (Literal["utf-8", "pkl"]):
                Encoding of keys. Defaults to "utf-8".
            value_encoding (Literal["utf-8", "pkl"]):
                Encoding of values. Defaults to "pkl".

        Raises:
            KeyError:
                If a non-existent key is requested.
        """
        self._lmdb_env = lmdb_env
        self._prefix = prefix + separator
        self._key_encoding = key_encoding
        self._value_encoding = value_encoding

        with self._lmdb_env.begin() as transaction, transaction.cursor() as cursor:
            # Collect all keys
            encoded_prefix = prefix.encode(self._key_encoding)
            # Assign the number of keys so don't have to store all keys in memory
            self._n_keys = len(
                [key for key, _ in cursor if key.startswith(encoded_prefix)]
            )

    def _decode_key(self, key):
        encoded_prefix = self._prefix.encode(self._key_encoding)
        return key[len(encoded_prefix) :].decode(self._key_encoding)

    def __iter__(self):
        "Use an iterative method to not have to store all keys in memory."
        encoded_prefix = self._prefix.encode(self._key_encoding)
        with self._lmdb_env.begin() as txn, txn.cursor() as cursor:
            # Seek to the first key >= prefix
            if cursor.set_range(encoded_prefix):
                while True:
                    current_key = cursor.key()
                    if not current_key.startswith(encoded_prefix):
                        break
                    # convert current_key into the user-facing key
                    # e.g. remove the prefix and decode if needed
                    user_key = self._decode_key(current_key)
                    yield user_key
                    if not cursor.next():
                        break

    def __len__(self):
        return self._n_keys

    def __getitem__(self, key):
        with self._lmdb_env.begin() as transaction:
            key_bytes = f"{self._prefix}{key}".encode(self._key_encoding)
            value_bytes = transaction.get(key_bytes)
            if not value_bytes:
                raise KeyError(key)
            else:
                if self._value_encoding == "pkl":
                    return pkl.loads(value_bytes)
                else:
                    return json.loads(value_bytes.decode(self._value_encoding))
