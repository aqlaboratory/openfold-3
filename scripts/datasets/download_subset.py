#!/usr/bin/env python3
"""Download the files an existing PDB subset cache references.

Requires training_cache_with_templates_subset_{train_size}.json /
validation_cache_with_templates_subset_{val_size}.json to already exist --
run generate_subset_cache.py first if they don't.

Downloads target structures, alignment arrays, and templates into
--output-dir, then reference-mol SDFs for only the CCD codes actually
present in the downloaded structures (not the full ~68k-file reference_mols
directory) -- see scripts/datasets/README.md for why.

Usage:
    python download_subset.py
    python download_subset.py --verify           # check completeness, don't download
    python download_subset.py --output-dir /data/foo
"""

import argparse
import sys
from pathlib import Path

from pdb_subset_helpers import (
    AMINO_ACID_CCD_CODES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_CACHE,
    DEFAULT_VAL_CACHE,
    STANDARD_NUCLEOTIDE_CCD_CODES,
    build_reference_mol_manifest,
    build_structure_manifest,
    download_manifest,
    extract_ids_from_cache,
    scan_structure_residue_names,
    structure_file_path,
    subset_cache_path,
    verify_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="Only verify, don't download"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel download threads"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to download files into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train-size", type=int, default=8, help="Train subset size (default: 8)"
    )
    parser.add_argument(
        "--val-size", type=int, default=4, help="Validation subset size (default: 4)"
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=DEFAULT_TRAIN_CACHE,
        help="Full training cache the subset was sampled from (for naming only)",
    )
    parser.add_argument(
        "--val-cache",
        type=Path,
        default=DEFAULT_VAL_CACHE,
        help="Full validation cache the subset was sampled from (for naming only)",
    )
    args = parser.parse_args()
    local_root = args.output_dir

    cache_files = {
        "train": subset_cache_path(args.train_cache, args.train_size),
        "val": subset_cache_path(args.val_cache, args.val_size),
    }
    missing = [str(p) for p in cache_files.values() if not p.exists()]
    if missing:
        sys.exit(
            f"Subset cache(s) not found: {missing}. Run generate_subset_cache.py first."
        )

    # Phase 1: structures, alignment arrays, templates (everything except
    # reference mols, which depend on what's actually in the structures).
    structure_manifest = []
    all_pdb_ids = set()
    cache_declared_ref_mols = set()

    for split, cache_path in cache_files.items():
        template_cache_subdir = (
            "train_template_cache" if split == "train" else "val_template_cache"
        )
        ids = extract_ids_from_cache(cache_path)
        manifest = build_structure_manifest(ids, template_cache_subdir, local_root)

        print(f"\n{split} ({cache_path.name}):")
        print(f"  PDB IDs: {len(ids['pdb_ids'])}")
        print(f"  Alignment rep IDs: {len(ids['alignment_rep_ids'])}")
        print(f"  Template IDs: {len(ids['template_ids'])}")
        print(f"  Files to download: {len(manifest)}")

        structure_manifest.extend(manifest)
        all_pdb_ids |= ids["pdb_ids"]
        cache_declared_ref_mols |= ids["reference_mol_ids"]

    # Deduplicate (alignment arrays shared between train/val)
    seen = set()
    deduped = []
    for s3_key, local_path in structure_manifest:
        if s3_key not in seen:
            seen.add(s3_key)
            deduped.append((s3_key, local_path))
    structure_manifest = deduped
    print(
        f"\nTotal unique structure/alignment/template files: {len(structure_manifest)}"
    )

    structure_paths = [
        structure_file_path(pdb_id, local_root) for pdb_id in all_pdb_ids
    ]

    if args.verify:
        missing = verify_manifest(structure_manifest)

        if all(p.exists() for p in structure_paths):
            ref_mol_ids = (
                scan_structure_residue_names(structure_paths)
                | cache_declared_ref_mols
                | AMINO_ACID_CCD_CODES
                | STANDARD_NUCLEOTIDE_CCD_CODES
            )
            ref_manifest = build_reference_mol_manifest(ref_mol_ids, local_root)
            print(f"Reference mol IDs (from structure scan): {len(ref_mol_ids)}")
            missing += verify_manifest(ref_manifest)
        else:
            print(
                "NOTE: not all target structures are downloaded yet, so "
                "reference-mol completeness can't be verified (which ones "
                "are needed depends on the actual residue composition of "
                "the downloaded structures)."
            )

        if missing:
            print(f"\nMISSING {len(missing)} files:")
            for m in missing[:20]:
                print(f"  {m}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
            sys.exit(1)
        else:
            print("\nAll files present!")
            sys.exit(0)

    counts = download_manifest(structure_manifest, workers=args.workers)
    print(
        f"\nStructures/alignments/templates: {counts['downloaded']} downloaded, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )

    # Phase 2: reference mols, determined from the actual residue composition
    # of the structures just downloaded (plus the cache-declared ligand ids
    # and the standard amino acid / nucleotide codes as a safety net).
    ref_mol_ids = (
        scan_structure_residue_names(structure_paths)
        | cache_declared_ref_mols
        | AMINO_ACID_CCD_CODES
        | STANDARD_NUCLEOTIDE_CCD_CODES
    )
    print(f"\nReference mol IDs (from structure scan): {len(ref_mol_ids)}")
    ref_manifest = build_reference_mol_manifest(ref_mol_ids, local_root)
    ref_counts = download_manifest(ref_manifest, workers=args.workers)
    print(
        f"Reference mols: {ref_counts['downloaded']} downloaded, "
        f"{ref_counts['skipped']} skipped, {ref_counts['failed']} failed"
    )

    total_downloaded = counts["downloaded"] + ref_counts["downloaded"]
    total_skipped = counts["skipped"] + ref_counts["skipped"]
    total_failed = counts["failed"] + ref_counts["failed"]
    print(
        f"\nDone: {total_downloaded} downloaded, "
        f"{total_skipped} skipped, {total_failed} failed"
    )


if __name__ == "__main__":
    main()
