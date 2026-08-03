"""Convert preprocessed structure NPZ files into smooth-lDDT benchmark cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

NUCLEOTIDE_MOLECULE_TYPE_IDS = (1, 2)


def _spatial_crop(
    coordinates: np.ndarray,
    molecule_types: np.ndarray,
    max_atoms: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_atoms is None or coordinates.shape[0] <= max_atoms:
        return coordinates, molecule_types

    centroid = coordinates.mean(axis=0)
    center_index = np.square(coordinates - centroid).sum(axis=-1).argmin()
    tree = cKDTree(coordinates)
    _, crop_indices = tree.query(coordinates[center_index], k=max_atoms)
    crop_indices = np.sort(np.asarray(crop_indices))
    return coordinates[crop_indices], molecule_types[crop_indices]


def generate_structure_case(
    structure_file: Path,
    max_atoms: int | None,
    noise_levels: tuple[float, ...],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]] | None:
    with np.load(structure_file, allow_pickle=False) as structure:
        if "coord" not in structure or "molecule_type_id" not in structure:
            return None
        coordinates = structure["coord"]
        molecule_types = structure["molecule_type_id"]

    resolved = np.isfinite(coordinates).all(axis=-1)
    coordinates = coordinates[resolved].astype(np.float32, copy=False)
    molecule_types = molecule_types[resolved]
    if coordinates.shape[0] < 2:
        return None

    original_n_atom = coordinates.shape[0]
    coordinates, molecule_types = _spatial_crop(coordinates, molecule_types, max_atoms)
    rng = np.random.default_rng(seed)
    predictions = np.stack(
        [
            coordinates + rng.normal(0.0, noise, coordinates.shape).astype(np.float32)
            for noise in noise_levels
        ]
    )
    is_nucleotide = np.isin(molecule_types, NUCLEOTIDE_MOLECULE_TYPE_IDS)
    atom_mask = np.ones(coordinates.shape[0], dtype=np.bool_)
    data = {
        "x_gt": coordinates,
        "x_pred": predictions,
        "is_nucleotide": is_nucleotide,
        "atom_mask": atom_mask,
        "loss_atom_mask": atom_mask.copy(),
        "noise_levels": np.asarray(noise_levels, dtype=np.float32),
    }
    metadata = {
        "profile": (
            "pulled_structure_full"
            if max_atoms is None
            else "pulled_structure_spatial_crop"
        ),
        "structure_id": structure_file.stem,
        "n_atom": coordinates.shape[0],
        "original_n_atom": original_n_atom,
        "n_nucleotide": int(is_nucleotide.sum()),
        "seed": seed,
        "source_file": str(structure_file),
    }
    return data, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        help="Only generate structures listed in this dataset cache.",
    )
    parser.add_argument("--max-atoms", type=int, default=3072)
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Keep every resolved atom instead of applying --max-atoms.",
    )
    parser.add_argument("--noise", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not args.no_crop and args.max_atoms < 2:
        parser.error("--max-atoms must be at least 2")
    max_atoms = None if args.no_crop else args.max_atoms

    structure_files = sorted(args.structure_dir.rglob("*.npz"))
    if not structure_files:
        parser.error(f"No NPZ files found under {args.structure_dir}")

    if args.dataset_cache is not None:
        cache = json.loads(args.dataset_cache.read_text())
        try:
            selected_ids = set(cache["structure_data"])
        except (KeyError, TypeError) as error:
            parser.error(
                f"{args.dataset_cache} does not contain a structure_data mapping: "
                f"{error}"
            )
        structure_files_by_id = {path.stem: path for path in structure_files}
        missing_ids = selected_ids - structure_files_by_id.keys()
        if missing_ids:
            parser.error(
                "Missing selected structure files: " + ", ".join(sorted(missing_ids))
            )
        structure_files = [
            structure_files_by_id[pdb_id] for pdb_id in sorted(selected_ids)
        ]

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output dir: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for structure_index, structure_file in enumerate(structure_files):
        generated = generate_structure_case(
            structure_file=structure_file,
            max_atoms=max_atoms,
            noise_levels=tuple(args.noise),
            seed=args.seed + structure_index,
        )
        if generated is None:
            continue
        data, metadata = generated
        suffix = "full" if args.no_crop else "spatial_crop"
        output_name = f"{structure_file.stem}_{suffix}.npz"
        np.savez_compressed(args.output_dir / output_name, **data)
        manifest.append({"file": output_name, **metadata})

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"cases": len(manifest), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
