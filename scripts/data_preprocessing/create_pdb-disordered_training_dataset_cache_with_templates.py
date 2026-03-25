#!/usr/bin/env python3
"""
Filter a disordered metadata cache against a parent training cache and write a
new (clustered) dataset cache JSON.

Usage example:
  python filter_disordered_cache.py \
    --metadata-cache /path/to/metadata_cache_all.json \
    --parent-cache /path/to/training_cache_with_templates.json \
    --out /path/to/output_cache.json \
    --gdt-threshold 0.6 \
    --clash-distance-threshold 1.1 \
    --name disordered-pdb
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from tqdm import tqdm

from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    DisorderedPreprocessingDataCache,
)


@click.command(context_settings={"show_default": True})
@click.option(
    "--metadata-cache",
    "metadata_cache_file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to disordered metadata cache JSON (e.g. metadata_cache_all.json).",
)
@click.option(
    "--parent-cache",
    "parent_dataset_cache_file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help=(
        "Path to parent training cache JSON (e.g. training_cache_with_templates.json)."
    ),
)
@click.option(
    "--out",
    "out_file",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Output path for the filtered dataset cache JSON.",
)
@click.option("--gdt-threshold", type=float, default=0.6, help="Minimum GDT to keep.")
@click.option(
    "--clash-distance-threshold",
    type=str,
    default="1.1",
    help=(
        "Distance threshold key used in distance_clash_map "
        '(often stored as a string, e.g. "1.1").'
    ),
)
@click.option("--name", type=str, default="disordered-pdb", help="Dataset cache name.")
@click.option(
    "--quiet",
    is_flag=True,
    help="Disable tqdm progress bar.",
)
@click.option(
    "--print-counts-json",
    is_flag=True,
    help="Print final counts as JSON (useful for logging).",
)
def main(
    metadata_cache_file: Path,
    parent_dataset_cache_file: Path,
    out_file: Path,
    gdt_threshold: float,
    clash_distance_threshold: str,
    name: str,
    quiet: bool,
    print_counts_json: bool,
) -> None:
    # Load caches
    metadata_cache = DisorderedPreprocessingDataCache.from_json(metadata_cache_file)
    parent_dataset_cache = read_datacache(parent_dataset_cache_file)

    # Track output structure_data and counts
    structure_data = {}
    counts = {
        "total": len(metadata_cache.structure_data),
        "fail_parent": 0,
        "fail_disordered": 0,
        "fail_entry_overlap": 0,
        "fail_chain_overlap": 0,
        "fail_gdt": 0,
        "fail_clash": 0,
        "final": 0,
    }

    iterator = metadata_cache.structure_data.items()
    if not quiet:
        iterator = tqdm(
            iterator,
            desc="Filtering disordered metadata cache",
            total=len(metadata_cache.structure_data),
        )

    for pdb_id, structure_data_entry in iterator:
        # Core success filter from disordered metadata
        if structure_data_entry.status != "success":
            # Keeping your original behavior/message (even though "parent" wording is a
            # bit confusing).
            click.echo(
                f"Fail parent cache status for {pdb_id}: {structure_data_entry.status}",
                err=True,
            )
            counts["fail_parent"] += 1
            continue

        # Secondary success filter from disordered cache
        if (
            (structure_data_entry.gdt is None)
            or (structure_data_entry.chain_map is None)
            or (structure_data_entry.distance_clash_map is None)
        ):
            counts["fail_disordered"] += 1
            continue

        # Entry overlap filter
        if pdb_id not in parent_dataset_cache.structure_data:
            counts["fail_entry_overlap"] += 1
            continue

        # Chain overlap filter
        chain_set = {
            (chain_id, chain_data.label_asym_id)
            for chain_id, chain_data in structure_data_entry.chains.items()
        }
        parent_chain_set = {
            (chain_id, chain_data.label_asym_id)
            for chain_id, chain_data in parent_dataset_cache.structure_data[
                pdb_id
            ].chains.items()
        }
        if chain_set != parent_chain_set:
            counts["fail_chain_overlap"] += 1
            continue

        # GDT filter
        if structure_data_entry.gdt < gdt_threshold:
            counts["fail_gdt"] += 1
            continue

        # Clash filter
        if clash_distance_threshold not in structure_data_entry.distance_clash_map:
            available = sorted(structure_data_entry.distance_clash_map.keys())
            raise click.ClickException(
                f'clash-distance-threshold "{clash_distance_threshold}" not found for '
                f"{pdb_id}. "
                "Available keys include: "
                f"{available[:20]}{'...' if len(available) > 20 else ''}"
            )

        if structure_data_entry.distance_clash_map[clash_distance_threshold]:
            counts["fail_clash"] += 1
            continue

        # Keep the parent cache entry (as in your original script)
        structure_data[pdb_id] = parent_dataset_cache.structure_data[pdb_id]

    dataset_cache = ClusteredDatasetCache(
        name=name,
        structure_data=structure_data,
        reference_molecule_data=parent_dataset_cache.reference_molecule_data,
    )
    counts["final"] = len(dataset_cache.structure_data)

    # Write output cache
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_datacache_to_json(dataset_cache, out_file)

    # Print counts
    if print_counts_json:
        click.echo(json.dumps(counts, indent=2, sort_keys=True))
    else:
        click.echo(str(counts))


if __name__ == "__main__":
    main()
