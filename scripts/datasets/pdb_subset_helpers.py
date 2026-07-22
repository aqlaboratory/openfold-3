"""Helpers for building and downloading subsets of the PDB training set.

Three families of helpers live here:
  - subset cache creation: enumerate_structure_ids / stream_subset / write_subset /
    sample_subset_cache, used to carve a smaller structure_data cache out of a full
    training/validation cache without loading the whole file into memory.
  - S3 download: extract_ids_from_cache / build_structure_manifest /
    build_reference_mol_manifest / scan_structure_residue_names /
    download_manifest / verify_manifest, used to fetch only the files
    referenced by a (sub)cache.
  - runner yaml generation: build_runner_yaml_config / write_runner_yaml, used to
    produce a run_openfold training config pointed at a downloaded subset.
"""

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import boto3
import ijson
import numpy as np
import yaml
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "openfold3-data"
S3_PREFIX = "pdb_training_set"

ROOT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "pdb_training_set"
DEFAULT_TRAIN_CACHE = ROOT_DIR / "training_cache_with_templates.json"
DEFAULT_VAL_CACHE = ROOT_DIR / "validation_cache_with_templates.json"
DEFAULT_RUNNER_YAML = ROOT_DIR / "train_pdb_subset.yaml"

AMINO_ACID_CCD_CODES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

STANDARD_NUCLEOTIDE_CCD_CODES = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}

# Full (un-sampled) dataset caches, keyed by filename, available on S3.
FULL_CACHE_S3_KEYS = {
    "training_cache_with_templates.json": (
        f"{S3_PREFIX}/dataset_caches/training_cache_with_templates.json"
    ),
    "validation_cache_with_templates.json": (
        f"{S3_PREFIX}/dataset_caches/validation_cache_with_templates.json"
    ),
}


# --------------------------------------------------------------------------
# Subset cache creation
# --------------------------------------------------------------------------


def download_full_cache(local_path: Path) -> None:
    """Ensure a full (un-sampled) dataset cache is present locally.

    No-ops if `local_path` already exists. Otherwise downloads it from its
    known S3 location (looked up by filename); raises if the filename isn't
    one of the known full caches.
    """
    if local_path.exists():
        return

    s3_key = FULL_CACHE_S3_KEYS.get(local_path.name)
    if s3_key is None:
        raise FileNotFoundError(
            f"{local_path} not found locally, and '{local_path.name}' is not a "
            f"known full cache on S3. Known caches: {sorted(FULL_CACHE_S3_KEYS)}"
        )

    print(
        f"{local_path.name} not found locally, "
        f"downloading from s3://{BUCKET}/{s3_key} ..."
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file(BUCKET, s3_key, str(local_path))

    # These cache files contain bare `NaN` literals (e.g. for structures with
    # no resolution data), which is invalid per RFC 8259 and breaks ijson's
    # strict streaming parser. Sanitize in place to `null`.
    text = local_path.read_text()
    local_path.write_text(re.sub(r"\bNaN\b", "null", text))


def subset_cache_path(full_cache: Path, size: int) -> Path:
    """Path a size-N subset cache for `full_cache` would live at.

    Shared by generate_subset_cache.py (writes it) and download_subset.py
    (reads it) so the naming convention only lives in one place.
    """
    return ROOT_DIR / f"{full_cache.stem}_subset_{size}.json"


def enumerate_structure_ids(path: Path) -> list[str]:
    """Stream top-level keys under `structure_data` without loading the file."""
    ids = []
    with open(path, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "structure_data" and event == "map_key":
                ids.append(value)
    return ids


def stream_subset(path: Path, selected_ids: set[str]) -> tuple[dict, dict]:
    """Return (metadata, structure_data_subset) streamed from a cache file."""
    metadata = {}
    with open(path, "rb") as f:
        for key, value in ijson.kvitems(f, ""):
            if key != "structure_data":
                metadata[key] = value

    selected_data = {}
    with open(path, "rb") as f:
        for pdb_id, data in ijson.kvitems(f, "structure_data"):
            if pdb_id in selected_ids:
                selected_data[pdb_id] = data
                if len(selected_data) == len(selected_ids):
                    break
    return metadata, selected_data


class _DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def write_subset(out_path: Path, metadata: dict, structure_data: dict) -> None:
    """Write a subset cache (metadata + structure_data) to out_path."""
    payload = {**metadata, "structure_data": structure_data}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=_DecimalEncoder)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Wrote {out_path} ({size_mb:.1f} MB, {len(structure_data)} structures)")


def sample_subset_cache(
    input_cache: Path,
    sizes: list[int],
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Randomly sample nested subsets of `input_cache` and write one file per size.

    Sizes are nested: a single random sample of size `max(sizes)` is drawn, and
    each smaller size is a prefix of that sample (e.g. sizes=[128, 256, 512] means
    the 128 subset is contained in the 256 subset, which is contained in the 512
    subset). Output files are named `{input_cache.stem}_subset_{size}.json`.
    """
    pdb_ids = enumerate_structure_ids(input_cache)
    print(f"Total PDB IDs in {input_cache.name}: {len(pdb_ids)}")

    max_size = max(sizes)
    rng = random.Random(seed)
    sampled = rng.sample(pdb_ids, max_size)
    # Keep subsets as sorted lists (not sets) so output ordering is deterministic
    # across runs/processes, independent of Python's per-process hash randomization.
    subsets = {size: sorted(sampled[:size]) for size in sorted(sizes)}

    metadata, largest_data = stream_subset(input_cache, set(subsets[max_size]))

    stem = input_cache.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    for size, selected_ids in subsets.items():
        subset_metadata = {
            **metadata,
            "name": f"{metadata.get('name', stem)}-subset-{size}",
        }
        write_subset(
            output_dir / f"{stem}_subset_{size}.json",
            subset_metadata,
            {pid: largest_data[pid] for pid in selected_ids},
        )


# --------------------------------------------------------------------------
# S3 download
# --------------------------------------------------------------------------


def extract_ids_from_cache(cache_path: Path) -> dict:
    """Extract all file IDs needed from a dataset cache JSON."""
    with open(cache_path) as f:
        cache = json.load(f)

    pdb_ids = set()
    alignment_rep_ids = set()
    template_ids = set()
    reference_mol_ids = set()

    for pdb_id, entry in cache["structure_data"].items():
        pdb_ids.add(pdb_id)
        for chain_data in entry["chains"].values():
            if rep_id := chain_data.get("alignment_representative_id"):
                alignment_rep_ids.add(rep_id)
            for tmpl_id in chain_data.get("template_ids") or []:
                template_ids.add(tmpl_id)
            if ref_mol := chain_data.get("reference_mol_id"):
                reference_mol_ids.add(ref_mol)

    return {
        "pdb_ids": pdb_ids,
        "alignment_rep_ids": alignment_rep_ids,
        "template_ids": template_ids,
        "reference_mol_ids": reference_mol_ids,
    }


def structure_file_path(pdb_id: str, local_root: Path) -> Path:
    """Local path of a target structure npz, matching build_structure_manifest."""
    return (
        local_root
        / "preprocessed_pdb_data"
        / "standard"
        / "structure_files"
        / pdb_id
        / f"{pdb_id}.npz"
    )


def build_structure_manifest(
    ids: dict, template_cache_subdir: str, local_root: Path
) -> list[tuple[str, Path]]:
    """Build list of (s3_key, local_path) tuples for everything except reference
    mols: target structures, alignment arrays, and templates.

    Reference mols are handled separately (see build_reference_mol_manifest)
    because which ones are needed can only be determined by inspecting the
    residue composition of the downloaded target structures themselves --
    the dataset cache's `reference_mol_id` is only set for standalone ligand
    chains, not for non-standard/modified residues embedded within a
    protein/RNA/DNA chain (e.g. a modified cysteine like CME), which also
    need their own reference conformer at train time.
    """
    manifest = []

    # Target structures: {pdb_id}/{pdb_id}.npz
    for pdb_id in ids["pdb_ids"]:
        s3_key = f"{S3_PREFIX}/preprocessed_pdb_data/standard/structure_files/{pdb_id}/{pdb_id}.npz"  # noqa: E501
        manifest.append((s3_key, structure_file_path(pdb_id, local_root)))

    # Alignment arrays: {rep_id}.npz
    for rep_id in ids["alignment_rep_ids"]:
        s3_key = f"{S3_PREFIX}/alignment_arrays/{rep_id}.npz"
        local = local_root / "alignment_arrays" / f"{rep_id}.npz"
        manifest.append((s3_key, local))

    # Template cache: {rep_id}.npz
    for rep_id in ids["alignment_rep_ids"]:
        s3_key = f"{S3_PREFIX}/templates/{template_cache_subdir}/{rep_id}.npz"
        local = local_root / "templates" / template_cache_subdir / f"{rep_id}.npz"
        manifest.append((s3_key, local))

    # Template structure arrays: {tmpl_pdb}/{tmpl_id}.npz
    for tmpl_id in ids["template_ids"]:
        tmpl_pdb = tmpl_id.split("_")[0]
        s3_key = (
            f"{S3_PREFIX}/templates/template_structure_arrays/{tmpl_pdb}/{tmpl_id}.npz"
        )
        local = (
            local_root
            / "templates"
            / "template_structure_arrays"
            / tmpl_pdb
            / f"{tmpl_id}.npz"
        )
        manifest.append((s3_key, local))

    # Also grab chain_id_to_moltype.npz for each template PDB
    seen_tmpl_pdbs = {tmpl_id.split("_")[0] for tmpl_id in ids["template_ids"]}
    for tmpl_pdb in seen_tmpl_pdbs:
        s3_key = f"{S3_PREFIX}/templates/template_structure_arrays/{tmpl_pdb}/chain_id_to_moltype.npz"  # noqa: E501
        local = (
            local_root
            / "templates"
            / "template_structure_arrays"
            / tmpl_pdb
            / "chain_id_to_moltype.npz"
        )
        manifest.append((s3_key, local))

    return manifest


def build_reference_mol_manifest(
    ref_mol_ids: set[str], local_root: Path
) -> list[tuple[str, Path]]:
    """Build list of (s3_key, local_path) tuples for reference mol SDFs."""
    manifest = []
    for ccd in sorted(ref_mol_ids):
        s3_key = f"{S3_PREFIX}/preprocessed_pdb_data/standard/reference_mols/{ccd}.sdf"
        local = (
            local_root
            / "preprocessed_pdb_data"
            / "standard"
            / "reference_mols"
            / f"{ccd}.sdf"
        )
        manifest.append((s3_key, local))
    return manifest


def scan_structure_residue_names(structure_paths: list[Path]) -> set[str]:
    """Return the union of unique CCD residue codes actually present across
    the given (already-downloaded) structure npz files.

    This is the ground truth for which reference mol SDFs are needed --
    covers both standalone ligands and non-standard/modified residues
    embedded within polymer chains. Missing files are skipped.
    """
    names = set()
    for path in structure_paths:
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        names.update(str(n) for n in np.unique(data["res_name"]))
    return names


def download_file(s3_client, s3_key: str, local_path: Path) -> tuple[str, str]:
    """Download a single file. Returns (s3_key, status)."""
    if local_path.exists():
        return (s3_key, "skipped")
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(BUCKET, s3_key, str(local_path))
        return (s3_key, "downloaded")
    except Exception as e:
        return (s3_key, f"FAILED: {e}")


def verify_manifest(manifest: list[tuple[str, Path]]) -> list[str]:
    """Check which files are missing. Returns the missing s3 keys."""
    missing = []
    for s3_key, local_path in manifest:
        if not local_path.exists():
            missing.append(s3_key)
    return missing


def download_manifest(
    manifest: list[tuple[str, Path]], workers: int = 8
) -> dict[str, int]:
    """Download every file in the manifest with a thread pool.

    Returns counts of downloaded/skipped/failed files.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, s3, s3_key, local_path): s3_key
            for s3_key, local_path in manifest
        }
        for i, future in enumerate(as_completed(futures), 1):
            s3_key, status = future.result()
            if status == "downloaded":
                counts["downloaded"] += 1
            elif status == "skipped":
                counts["skipped"] += 1
            else:
                counts["failed"] += 1
                print(f"  {status}: {s3_key}")

            if i % 100 == 0 or i == len(futures):
                print(
                    f"  Progress: {i}/{len(futures)} "
                    f"(downloaded={counts['downloaded']}, "
                    f"skipped={counts['skipped']}, "
                    f"failed={counts['failed']})"
                )

    return counts


# --------------------------------------------------------------------------
# Runner yaml generation
# --------------------------------------------------------------------------


def _dataset_paths_entry(
    cache_file: Path, local_root: Path, template_cache_subdir: str
) -> dict:
    return {
        "alignments_directory": "none",
        "alignment_db_directory": "none",
        "alignment_array_directory": str(local_root / "alignment_arrays"),
        "dataset_cache_file": str(cache_file),
        "target_structures_directory": str(
            local_root
            / "preprocessed_pdb_data"
            / "standard"
            / "structure_files"
        ),
        "target_structure_file_format": "npz",
        "reference_molecule_directory": str(
            local_root / "preprocessed_pdb_data" / "standard" / "reference_mols"
        ),
        "template_cache_directory": str(
            local_root / "templates" / template_cache_subdir
        ),
        "template_structure_array_directory": str(
            local_root / "templates" / "template_structure_arrays"
        ),
        "template_structures_directory": "none",
        "template_file_format": "npz",
        "ccd_file": None,
    }


def build_runner_yaml_config(cache_files: dict[str, Path], local_root: Path) -> dict:
    """Build a run_openfold training config for a downloaded PDB subset.

    `cache_files` must have "train" and "val" keys pointing at the respective
    dataset cache jsons. `local_root` is the directory the referenced files
    were downloaded into (i.e. download_subset.py's --output-dir), matching
    the layout produced by build_manifest.
    """
    return {
        "experiment_settings": {
            "mode": "train",
            "output_dir": "./test_train_output",
            "seed": 42,
            "restart_checkpoint_path": "last",
        },
        "data_module_args": {
            "batch_size": 1,
            "num_workers": 4,
            "epoch_len": 32,
        },
        "logging_config": {
            "log_lr": False,
            "wandb_config": None,
        },
        "pl_trainer_args": {
            "devices": 1,
            "num_nodes": 1,
            "precision": "bf16-mixed",
            "max_epochs": 2,
            "mpi_plugin": False,
            "deepspeed_config_path": None,
        },
        "model_update": {
            "presets": ["train"],
            "custom": {
                "settings": {
                    "model_selection_weight_scheme": "fine_tuning",
                    "memory": {
                        "train": {
                            "msa_module": {"swiglu_seq_chunk_size": 1024},
                            "use_cueq_triangle_kernels": False,
                            "use_deepspeed_evo_attention": True,
                        },
                    },
                },
                "architecture": {
                    "shared": {
                        "use_confidence_emb_prob": 0.8,
                        "diffusion": {"use_conditioning_prob": 0.8},
                    },
                    "loss_module": {
                        "diffusion": {"chunk_size": 2},
                    },
                },
            },
        },
        "dataset_configs": {
            "train": {
                "weighted-pdb": {
                    "dataset_class": "WeightedPDBDataset",
                    "weight": 0.5,
                    "config": {
                        "debug_mode": True,
                        "template": {"n_templates": 4, "take_top_k": False},
                        "crop": {
                            "token_crop": {
                                "enabled": True,
                                "token_budget": 384,
                                "crop_weights": {
                                    "contiguous": 0.2,
                                    "spatial": 0.4,
                                    "spatial_interface": 0.4,
                                },
                            },
                            "chain_crop": {"enabled": True},
                        },
                    },
                },
            },
            "validation": {
                "val-weighted-pdb": {
                    "dataset_class": "ValidationPDBDataset",
                    "config": {
                        "debug_mode": True,
                        "msa": {"subsample_main": False},
                        "template": {"n_templates": 4, "take_top_k": True},
                        "crop": {"token_crop": {"enabled": False}},
                    },
                },
            },
        },
        "dataset_paths": {
            "weighted-pdb": _dataset_paths_entry(
                cache_files["train"], local_root, "train_template_cache"
            ),
            "val-weighted-pdb": _dataset_paths_entry(
                cache_files["val"], local_root, "val_template_cache"
            ),
        },
    }


def write_runner_yaml(
    output_path: Path, cache_files: dict[str, Path], local_root: Path
) -> None:
    """Write a run_openfold training runner yaml for a downloaded PDB subset."""
    config = build_runner_yaml_config(cache_files, local_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Auto-generated by generate_subset_cache.py -- do not edit by hand.\n")
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote runner yaml to {output_path}")
