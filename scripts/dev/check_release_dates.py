"""
Convenience script for sanity-checking release dates of PDB entries and templates in
training or validation dataset caches.
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from tqdm import tqdm

from openfold3.core.data.io.dataset_cache import read_datacache


def iso_to_date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def main():
    parser = argparse.ArgumentParser(
        description="Check PDB entry/template release dates against bounds"
    )
    parser.add_argument(
        "release_cache",
        type=Path,
        help="JSON file mapping PDB ID → ISO date or null",
    )
    parser.add_argument(
        "dataset_cache",
        type=Path,
        help="OpenFold3 dataset cache JSON",
    )
    parser.add_argument(
        "--min-entry-date",
        type=date.fromisoformat,
        default=None,
        help="ISO date; report entries older than this",
    )
    parser.add_argument(
        "--max-entry-date",
        type=date.fromisoformat,
        default=None,
        help="ISO date; report entries newer than this",
    )
    parser.add_argument(
        "--max-template-date",
        type=date.fromisoformat,
        default=None,
        help="ISO date; report templates newer than this",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional file to write logs to",
    )
    args = parser.parse_args()

    # ——— configure logging ———
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if args.log_file:
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.info("Loading release date cache…")
    raw = json.loads(args.release_cache.read_text())
    release_dates = {pid: iso_to_date(d) for pid, d in raw.items()}
    raw_keys = set(raw.keys())

    logger.info(f"Reading dataset cache {args.dataset_cache}…")
    dc = read_datacache(args.dataset_cache)
    struct = dc.structure_data
    entry_ids = set(struct.keys())

    # Collect all template IDs
    template_ids = {
        tid[:4]
        for entry in tqdm(struct.values(), desc="Collecting template IDs")
        for chain in entry.chains.values()
        if chain.template_ids
        for tid in chain.template_ids
    }

    missing_ids = set()
    tested_count = 0

    results: dict = {"dataset_cache_path": str(args.dataset_cache)}

    # Only check entries if the user passed at least one entry‐date flag
    if args.min_entry_date or args.max_entry_date:
        entries_too_old = []
        entries_too_new = []
        for pid in tqdm(sorted(entry_ids), desc="Checking entries"):
            tested_count += 1
            if pid not in raw_keys:
                missing_ids.add(pid)
                continue
            d = release_dates.get(pid)
            if not d:
                missing_ids.add(pid)
                continue
            if args.min_entry_date and d < args.min_entry_date:
                entries_too_old.append(pid)
            if args.max_entry_date and d > args.max_entry_date:
                entries_too_new.append(pid)

        if entries_too_new:
            results["entries_outside_max"] = sorted(entries_too_new)
            logger.warning(f"Entries newer than max-entry-date: {entries_too_new}")
        if args.min_entry_date and entries_too_old:
            results["entries_outside_min"] = sorted(entries_too_old)
            logger.warning(f"Entries older than min-entry-date: {entries_too_old}")

    # Only check templates if --max-template-date was provided
    if args.max_template_date:
        templates_too_new = []
        for pid in tqdm(sorted(template_ids), desc="Checking templates"):
            tested_count += 1
            if pid not in raw_keys:
                missing_ids.add(pid)
                continue
            d = release_dates.get(pid)
            if not d:
                missing_ids.add(pid)
                continue
            if d > args.max_template_date:
                templates_too_new.append(pid)

        if templates_too_new:
            results["templates_outside_max"] = sorted(templates_too_new)
            logger.warning(
                f"Templates newer than max-template-date: {templates_too_new}"
            )

    if missing_ids:
        logger.warning(
            f"{len(missing_ids)} PDB IDs not found in release date cache (ignored): "
            f"{sorted(missing_ids)}"
        )

    # ——— all-clear only if no date-filter violations at all ———
    if set(results.keys()) == {"dataset_cache_path"}:
        logger.info("All checks passed: no entries or templates outside date bounds.")

    logger.info(f"Tested {tested_count} PDB IDs for release dates in total")
    logger.info("Final results:\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
