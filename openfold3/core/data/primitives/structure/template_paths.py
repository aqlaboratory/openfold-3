"""Utilities for resolving template structure paths."""

from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm


def resolve_template_structure_path(
    template_structures_directory: Path,
    pdb_id: str,
    template_filename: str,
) -> Path:
    """Resolve a template in the flat directory or its sharded subdirectory."""
    flat_path = template_structures_directory / template_filename
    if flat_path.is_file():
        return flat_path

    if len(pdb_id) >= 3:
        sharded_path = template_structures_directory / pdb_id[-3:-1] / template_filename
        if sharded_path.is_file():
            return sharded_path

    raise FileNotFoundError(
        f"No template structure found for {pdb_id}: "
        f"checked {flat_path} and its sharded path"
    )


def iter_template_structure_paths(
    template_structures_directory: Path,
    template_file_format: str,
) -> Iterator[Path]:
    """Yield all template files for preprocessing from flat and sharded (two-character)
    paths.

    Unlike ``resolve_template_structure_path``, this function intentionally
    iterates the directory contents because preprocessing needs every template.
    """
    suffix = f".{template_file_format}"
    paths_by_name: dict[str, Path] = {}

    for path in tqdm(
        template_structures_directory.iterdir(),
        desc="Scanning template structures",
    ):
        if path.is_file() and path.name.endswith(suffix):
            paths_by_name[path.name] = path

    for shard_directory in tqdm(
        template_structures_directory.iterdir(),
        desc="Scanning template shards",
    ):
        if not shard_directory.is_dir() or len(shard_directory.name) != 2:
            continue
        for path in tqdm(
            shard_directory.iterdir(),
            desc=f"Scanning template shard {shard_directory.name}",
            leave=False,
        ):
            if path.is_file() and path.name.endswith(suffix):
                paths_by_name.setdefault(path.name, path)

    yield from (paths_by_name[name] for name in sorted(paths_by_name))
