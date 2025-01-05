import json
from pathlib import Path

import lmdb
from tqdm import tqdm

from openfold3.core.data.io.dataset_cache import (
    convert_dataclass_to_dict,
    read_datacache,
)


def convert_datacache_to_lmdb(
    dataset_cache_file: Path,
    lmdb_directory: Path,
    map_size: int = 2 * (1024**3),
    mode: str = "single-read",
    encoding: str = "utf-8",
) -> None:
    """Convert a JSON file to an LMDB file.

    Args:
        json_file (Path):
            The datacache JSON file to convert.
        lmdb_dir (Path):
            The LMDB dir to which the data and lock files are to be written.
        map_size (int):
            Size of the json file.
        mode (str):
            The mode to use to parse the json file. Can be one of 'single-read' or
            'iterative'. Use 'single-read' for small files and 'iterative' for large
            files.
        encoding (str):
            The encoding to use for the LMDB.
    """

    if mode not in ["single-read", "iterative"]:
        raise ValueError("Invalid mode. Must be one of 'single-read' or 'iterative'.")

    if mode == "single-read":
        dataset_cache = read_datacache(dataset_cache_file)

        lmdb_env = lmdb.open(lmdb_directory, map_size=map_size, subdir=True)

        with lmdb_env.begin(write=True) as transaction:
            print("1/4: Adding _type to the LMDB.")
            transaction.put(b"_type", json.dumps(dataset_cache._type).encode(encoding))
            print("2/4: Adding name to the LMDB.")
            transaction.put(b"name", json.dumps(dataset_cache.name).encode(encoding))

            # Store each entry in structure_data separately
            for sdata_key, sdata_value in tqdm(
                dataset_cache.structure_data.items(),
                desc="3/4: Adding structure_data to the LMDB",
                total=len(dataset_cache.structure_data),
            ):
                key_bytes = f"structure_data:{sdata_key}".encode(encoding)
                sdata_value_dict = convert_dataclass_to_dict(sdata_value)
                val_bytes = json.dumps(sdata_value_dict).encode(encoding)
                transaction.put(key_bytes, val_bytes)

            # Store each entry in reference_molecule_data separately
            for ref_mol_key, ref_mol_info in tqdm(
                dataset_cache.reference_molecule_data.items(),
                desc="4/4: Adding reference_molecule_data to the LMDB",
                total=len(dataset_cache.reference_molecule_data),
            ):
                key_bytes = f"reference_molecule_data:{ref_mol_key}".encode(encoding)
                ref_mol_info_dict = convert_dataclass_to_dict(ref_mol_info)
                val_bytes = json.dumps(ref_mol_info_dict).encode(encoding)
                transaction.put(key_bytes, val_bytes)

        lmdb_env.close()


# update with LMDB dict-like class
# def fetch_lmdb_entry(lmdb_directory: Path, pdb_id) -> None:
#     lmdb_env = lmdb.open(lmdb_directory, readonly=True, lock=False, subdir=True)
#     with lmdb_env.begin() as transaction:
#         # Retrieve the small top-level
#         type = transaction.get(b"_type")
#         if small_top_level_bytes:
#             small_data = json.loads(small_top_level_bytes.decode("utf-8"))
#             print("Small top-level data:", small_data)

#         # Retrieve one structure_data entry (for example, '6ouk')
#         val_bytes = transaction.get(b"structure_data:6ouk")
#         if val_bytes:
#             val = json.loads(val_bytes.decode("utf-8"))
#             print("structure_data:6ouk =>", val.keys())

#         # Retrieve a reference molecule entry (e.g. 'N7J')
#         val_bytes = transaction.get(b"reference_molecule_data:N7J")
#         if val_bytes:
#             val = json.loads(val_bytes.decode("utf-8"))
#             print("reference_molecule_data:N7J =>", val)

#     lmdb_env.close()
