"""Script that gets the release dates for all PDB IDs in an mmCIF directory."""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from biotite.structure.io.pdbx import CIFFile
from tqdm import tqdm

from openfold3.core.data.primitives.structure.metadata import (
    get_cif_block,
    get_release_date,
)


def fetch_release_date(pdb_dir: Path, pdb_id: str):
    """
    Return a datetime.date for pdb_id, or None on error/missing file.
    """
    cif_path = pdb_dir / f"{pdb_id}.cif"
    if not cif_path.exists():
        print(f"Warning: {pdb_id}.cif not found in {pdb_dir}")
        return None
    try:
        cif = CIFFile.read(cif_path)
        block = get_cif_block(cif)
        return get_release_date(block).date()
    except Exception as e:
        print(f"Error reading {pdb_id}: {e}")
        return None


def _worker(args):
    # Small wrapper since Pool only passes a single argument
    pdb_dir, pdb_id = args
    return pdb_id, fetch_release_date(pdb_dir, pdb_id)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a JSON cache mapping PDB ID → release date from a raw PDB directory."
        )
    )
    parser.add_argument(
        "pdb_dir",
        type=Path,
        help="Directory containing <pdb_id>.cif files",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output JSON file (will be overwritten)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes for I/O",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50,
        help="Chunksize to use with multiprocessing pool",
    )
    args = parser.parse_args()

    # Collect all PDB IDs
    pdb_ids = [p.stem for p in args.pdb_dir.glob("*.cif")]
    mapping: dict[str, str | None] = {}

    # Prepare argument tuples for the worker
    tasks = [(args.pdb_dir, pid) for pid in pdb_ids]

    # Parallel fetching with progress bar
    with Pool(processes=args.num_workers) as pool:
        for pid, dt in tqdm(
            pool.imap_unordered(_worker, tasks, chunksize=args.chunksize),
            total=len(tasks),
            desc="Fetching release dates",
        ):
            mapping[pid] = dt.isoformat() if dt is not None else None

    # Write results to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out_f:
        json.dump(mapping, out_f, indent=4)
    print(f"Written {len(mapping)} entries to {args.output}")


if __name__ == "__main__":
    main()
