"""Generate an MSA representatives FASTA file from preparsed NPZ alignment files.

Usage:
    python generate_representatives_from_preparsed.py \
        --npz-directory /path/to/npz/dir \
        --out-fasta /path/to/output.fasta \
        --ncores 16
"""

import json
import logging
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_PROTEIN_KEYS = {"cfdb_hits", "mgnify_hits", "uniprot_hits", "uniref90_hits"}
DEFAULT_RNA_KEYS = {
    "nt_hits",
    "rfam_hits",
    "rnacentral_hits",
    "nucleotide_collection_hits",
}


@dataclass
class NpzResult:
    """Result from processing a single NPZ file."""

    npz_file: str
    status: str  # "PASS", "WARN", "FAIL"
    reason: str | None
    msa_id: str | None
    moltype: str | None
    query_seq: str | None
    num_db_keys: int
    total_msa_depth: int  # sum of num_sequences across all matched DBs


def _fail(
    npz_path: Path,
    reason: str,
    *,
    moltype: str | None = None,
    num_db_keys: int = 0,
) -> NpzResult:
    return NpzResult(
        npz_file=str(npz_path),
        status="FAIL",
        reason=reason,
        msa_id=npz_path.stem,
        moltype=moltype,
        query_seq=None,
        num_db_keys=num_db_keys,
        total_msa_depth=0,
    )


def process_single_npz(
    npz_path: Path, protein_keys: set[str], rna_keys: set[str]
) -> NpzResult:
    """Process a single NPZ file: extract query sequence and detect moltype."""
    msa_id = npz_path.stem

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            keys_in_file = set(data.keys())
            # Classify moltype by which key set matches
            matched_protein = keys_in_file & protein_keys
            matched_rna = keys_in_file & rna_keys

            if matched_protein and matched_rna:
                return _fail(
                    npz_path,
                    f"Both protein and RNA keys detected. "
                    f"Protein: {sorted(matched_protein)}, "
                    f"RNA: {sorted(matched_rna)}",
                )
            elif matched_protein:
                moltype = "protein"
                expected = protein_keys
                matched = matched_protein
            elif matched_rna:
                moltype = "rna"
                expected = rna_keys
                matched = matched_rna
            else:
                return _fail(
                    npz_path,
                    f"No recognized database keys. "
                    f"Keys in file: {sorted(keys_in_file)}",
                )

            status, reason = "PASS", None
            missing = expected - keys_in_file
            if missing:
                status = "WARN"
                reason = f"Missing expected {moltype} DB keys: {sorted(missing)}"

            # Query sequence = row 0 of each MSA; verify consistency across DBs
            query_seqs = {}
            total_msa_depth = 0
            for db_key in matched:
                msa_array = data[db_key].item()["msa"]
                total_msa_depth += msa_array.shape[0]
                if msa_array.shape[0] > 0:
                    query_seqs[db_key] = "".join(msa_array[0])
                del msa_array  # free large array eagerly

        if not query_seqs:
            return _fail(
                npz_path,
                "All MSA arrays are empty",
                moltype=moltype,
                num_db_keys=len(matched),
            )

        unique_queries = set(query_seqs.values())
        if len(unique_queries) > 1:
            return _fail(
                npz_path,
                f"Inconsistent query sequences across DBs. "
                f"Per-DB lengths: { ({k: len(v) for k, v in query_seqs.items()}) }",
                moltype=moltype,
                num_db_keys=len(matched),
            )

        raw_query = next(iter(unique_queries))

        if "-" in raw_query:
            gap_msg = "Gap characters found in query sequence"
            reason = f"{reason}; {gap_msg}" if reason else gap_msg
            status = "WARN"
            raw_query = raw_query.replace("-", "")

        if not raw_query:
            return _fail(
                npz_path,
                "Query sequence empty after gap stripping",
                moltype=moltype,
                num_db_keys=len(matched),
            )

        return NpzResult(
            npz_file=str(npz_path),
            status=status,
            reason=reason,
            msa_id=msa_id,
            moltype=moltype,
            query_seq=raw_query,
            num_db_keys=len(matched),
            total_msa_depth=total_msa_depth,
        )

    except Exception as e:
        return _fail(npz_path, f"{type(e).__name__}: {e}")


class _NpzProcessor:
    """Callable wrapper so constant args are pickled once, not per-task."""

    def __init__(self, protein_keys: set[str], rna_keys: set[str]):
        self.protein_keys = protein_keys
        self.rna_keys = rna_keys

    def __call__(self, npz_path: Path) -> NpzResult:
        return process_single_npz(npz_path, self.protein_keys, self.rna_keys)


def deduplicate_results(results: list[NpzResult]) -> list[NpzResult]:
    """Deduplicate by (sequence, moltype), keeping the best entry.

    Tiebreakers (in order): more DB keys -> deeper total MSA -> keep existing.
    """
    best: dict[tuple[str, str], NpzResult] = {}

    for result in results:
        key = (result.query_seq, result.moltype)
        if key not in best:
            best[key] = result
            continue

        existing = best[key]
        if result.num_db_keys > existing.num_db_keys:
            replace = True
        elif result.num_db_keys < existing.num_db_keys:
            replace = False
        elif result.total_msa_depth > existing.total_msa_depth:
            replace = True
        else:
            replace = False

        if replace:
            logger.info(
                "Dedup: dropped %s in favor of %s (%d/%d vs %d/%d keys/depth)",
                existing.msa_id,
                result.msa_id,
                existing.num_db_keys,
                existing.total_msa_depth,
                result.num_db_keys,
                result.total_msa_depth,
            )
            best[key] = result

    return list(best.values())


@click.command()
@click.option(
    "--npz-directory",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing preparsed .npz alignment files.",
)
@click.option(
    "--out-fasta",
    required=True,
    type=click.Path(),
    help="Output path for the representatives FASTA file.",
)
@click.option(
    "--protein-keys",
    default=",".join(sorted(DEFAULT_PROTEIN_KEYS)),
    help="Comma-separated protein database key names.",
)
@click.option(
    "--rna-keys",
    default=",".join(sorted(DEFAULT_RNA_KEYS)),
    help="Comma-separated RNA database key names.",
)
@click.option("--ncores", default=1, type=int, help="Number of parallel workers.")
def main(
    npz_directory: str, out_fasta: str, protein_keys: str, rna_keys: str, ncores: int
) -> None:
    """Generate an MSA representatives FASTA from preparsed NPZ files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    npz_dir = Path(npz_directory)
    out_path = Path(out_fasta)

    pkeys = set(protein_keys.split(",")) if protein_keys else set()
    rkeys = set(rna_keys.split(",")) if rna_keys else set()
    if not pkeys and not rkeys:
        raise click.UsageError(
            "Must provide at least one of --protein-keys or --rna-keys"
        )

    logger.info("Protein keys: %s", sorted(pkeys))
    logger.info("RNA keys: %s", sorted(rkeys))

    npz_files = sorted(npz_dir.glob("*.npz"))
    logger.info("Found %d NPZ files in %s", len(npz_files), npz_dir)
    if not npz_files:
        logger.error("No NPZ files found. Exiting.")
        return

    # Process in parallel
    processor = _NpzProcessor(protein_keys=pkeys, rna_keys=rkeys)
    chunksize = min(64, max(1, len(npz_files) // (ncores * 4)))

    all_results: list[NpzResult] = []
    with mp.Pool(ncores) as pool:
        for result in tqdm(
            pool.imap_unordered(processor, npz_files, chunksize=chunksize),
            total=len(npz_files),
            desc="Processing NPZ files",
        ):
            all_results.append(result)

    successes, failures, warnings = [], [], []
    for r in all_results:
        if r.query_seq is None:
            failures.append(r)
        else:
            successes.append(r)
            if r.status == "WARN":
                warnings.append(r)
    logger.info(
        "Processing complete: %d passed, %d warnings, %d failed",
        len(successes),
        len(warnings),
        len(failures),
    )

    unique_results = deduplicate_results(successes)
    n_removed = len(successes) - len(unique_results)
    logger.info(
        "After dedup: %d unique entries (removed %d duplicates)",
        len(unique_results),
        n_removed,
    )

    # Write FASTA
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in sorted(unique_results, key=lambda x: x.msa_id):
            f.write(f">{r.msa_id}|{r.moltype}\n{r.query_seq}\n")
    logger.info("Wrote %d entries to %s", len(unique_results), out_path)

    # Write summary JSON
    summary_path = out_path.parent / "representative_processing_summary.json"
    summary = {
        "counts": {
            "total": len(all_results),
            "passed": len(successes),
            "warnings": len(warnings),
            "failed": len(failures),
            "duplicates_removed": n_removed,
            "unique_entries": len(unique_results),
        },
        "files": sorted(  # only non-PASS entries to keep JSON manageable at scale
            [asdict(r) for r in all_results if r.status != "PASS"],
            key=lambda x: x["npz_file"],
        ),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote processing summary to %s", summary_path)


if __name__ == "__main__":
    main()
