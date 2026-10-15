#!/usr/bin/env python3
"""Selection logic behind the smoke test's pinned validation set.

The training integration test's "smoke" case runs validation without cropping
(`crop.token_crop.enabled: false`, matching real validation), so a *randomly*
sampled validation subset can happen to include a huge structure -- we hit an
11+ minute stall on one uncropped structure with token_count=1985. Randomly
resampling with a token_count cap was tried and rejected (see
`git log -p -- scripts/datasets/pdb_subset_helpers.py` around this script's
introduction): it still means scanning/filtering over the full ~1700-structure
cache on every regeneration. Instead, this script is run *once* to pick a
small, fixed, modality-diverse set of PDB IDs, which then get pinned directly
in `pdb_subset_helpers.SMOKE_VALIDATION_PDB_IDS` -- no scanning/filtering at
subset-generation time afterward, just a direct lookup of known-good IDs.

Selection criteria, applied to the full `validation_cache_with_templates.json`:
  - token_count <= 300 (small enough that an uncropped validation pass is
    always cheap, regardless of what large structures exist elsewhere in the
    full cache).
  - One clean, non-overlapping representative per modality, so each pinned ID
    exercises a distinct code path rather than accidentally covering two at
    once:
      - dna: has a DNA chain, no RNA chain, no ligand chain.
      - rna: has an RNA chain, no DNA chain, no ligand chain.
      - protein_ligand: exactly one PROTEIN chain plus >=1 LIGAND chain(s),
        no DNA/RNA.
      - multimer: >=2 PROTEIN chains, no DNA/RNA/ligand.
  - Within each modality, the smallest (by token_count) candidate is picked.
  - Manually verified against S3 (see this script's `--verify` flag) that the
    target structure file exists for each pick. A missing *template* cache
    file is not disqualifying by itself -- it's the expected, valid state for
    a chain with no templates found (`template_ids: null` in the cache), not
    a gap; the RNA pick below is one such case.

Result (as of 2026-07-26, against validation_cache_with_templates.json):
  dna:             7ohe   (24 tokens,  DNA duplex, 2 chains)
  rna:             7kud   (13 tokens,  RNA, no templates found -- expected)
  protein_ligand:  7vus   (87 tokens,  1 protein chain + 3 ligand chains)
  multimer:        7fb8   (53 tokens,  protein homodimer)

If these ever need to change (e.g. one gets pulled from the upstream cache, or
the criteria change), rerun this script and update
`pdb_subset_helpers.SMOKE_VALIDATION_PDB_IDS` by hand -- it's a plain
dict, not derived from this script automatically.

Usage:
    python select_smoke_validation_set.py                    # rank candidates
    python select_smoke_validation_set.py --max-token-count 500
    python select_smoke_validation_set.py --verify 7ohe 7kud 7vus 7fb8
"""

import argparse
from pathlib import Path

import boto3
import ijson
from botocore import UNSIGNED
from botocore.config import Config
from pdb_subset_helpers import BUCKET, S3_PREFIX, default_target_dir

MODALITIES = ("dna", "rna", "protein_ligand", "multimer")


def classify(entry: dict) -> list[str]:
    """Which of MODALITIES `entry` is a clean, non-overlapping candidate for."""
    mol_types = [c.get("molecule_type") for c in entry.get("chains", {}).values()]
    has_dna = "DNA" in mol_types
    has_rna = "RNA" in mol_types
    has_ligand = "LIGAND" in mol_types
    protein_count = mol_types.count("PROTEIN")

    hits = []
    if has_dna and not has_rna and not has_ligand:
        hits.append("dna")
    if has_rna and not has_dna and not has_ligand:
        hits.append("rna")
    if has_ligand and protein_count == 1 and not has_dna and not has_rna:
        hits.append("protein_ligand")
    if protein_count >= 2 and not has_dna and not has_rna and not has_ligand:
        hits.append("multimer")
    return hits


def find_candidates(val_cache: Path, max_token_count: int) -> dict[str, list]:
    candidates = {m: [] for m in MODALITIES}
    with open(val_cache, "rb") as f:
        for pdb_id, entry in ijson.kvitems(f, "structure_data"):
            token_count = entry.get("token_count")
            if token_count is None or token_count > max_token_count:
                continue
            for modality in classify(entry):
                candidates[modality].append((token_count, pdb_id))
    for rows in candidates.values():
        rows.sort()
    return candidates


def verify_s3(val_cache: Path, pdb_ids: list[str]) -> None:
    """Check target-structure/alignment/template S3 keys for each pdb_id."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    def exists(key: str) -> bool:
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            return True
        except Exception:
            return False

    wanted = set(pdb_ids)
    entries = {}
    with open(val_cache, "rb") as f:
        for pdb_id, entry in ijson.kvitems(f, "structure_data"):
            if pdb_id in wanted:
                entries[pdb_id] = entry

    for pdb_id in pdb_ids:
        entry = entries.get(pdb_id)
        if entry is None:
            print(f"{pdb_id}: NOT FOUND in {val_cache.name}")
            continue
        struct_key = (
            f"{S3_PREFIX}/preprocessed_pdb_data/standard/structure_files/"
            f"{pdb_id}/{pdb_id}.npz"
        )
        print(f"{pdb_id}: structure_ok={exists(struct_key)}")
        for chain_id, chain in entry["chains"].items():
            rep = chain.get("alignment_representative_id")
            if not rep:
                continue
            aln_ok = exists(f"{S3_PREFIX}/alignment_arrays/{rep}.npz")
            tmpl_ok = exists(f"{S3_PREFIX}/templates/val_template_cache/{rep}.npz")
            has_templates = bool(chain.get("template_ids"))
            print(
                f"  chain {chain_id} ({rep}): alignment_ok={aln_ok} "
                f"template_ok={tmpl_ok} (template_ids present: {has_templates})"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-cache",
        type=Path,
        default=None,
        help=(
            "Full validation cache"
            "(default: <target-dir>/validation_cache_with_templates.json)"
        ),
    )
    parser.add_argument(
        "--max-token-count",
        type=int,
        default=300,
        help="Only rank candidates at or below this token_count (default: 300)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="How many ranked candidates to print per modality (default: 8)",
    )
    parser.add_argument(
        "--verify",
        nargs="+",
        metavar="PDB_ID",
        help="Instead of ranking, check S3 availability for these specific IDs",
    )
    args = parser.parse_args()

    val_cache = args.val_cache or (
        default_target_dir() / "validation_cache_with_templates.json"
    )

    if args.verify:
        verify_s3(val_cache, args.verify)
        return

    candidates = find_candidates(val_cache, args.max_token_count)
    for modality, rows in candidates.items():
        print(
            f"=== {modality}: {len(rows)} candidates "
            f"(<= {args.max_token_count} tokens) ==="
        )
        for token_count, pdb_id in rows[: args.top]:
            print(f"  {pdb_id:8s} tokens={token_count}")


if __name__ == "__main__":
    main()
