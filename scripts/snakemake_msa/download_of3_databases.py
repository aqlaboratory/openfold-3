import json
import logging
import subprocess as sp
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Final

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# S3 bucket configuration
S3_BUCKET: Final[str] = "openfold"
S3_PREFIX: Final[str] = "alignment_databases"

# Database names
JACKHMMER_DATABASES: Final[list[str]] = ["uniprot", "uniref90", "mgnify", "pdb_seqres"]
RNA_DATABASES: Final[list[str]] = ["rfam", "rnacentral", "nucleotide_collection"]
HHBLITS_DATABASES: Final[list[str]] = ["uniref30"]
BFD_DATABASE: Final[str] = "bfd"
CFDB_DATABASE: Final[str] = "cfdb"


def get_known_database_info() -> dict[str, str]:
    """Return mapping of archive names to their type (Protein or DNA/RNA)."""
    known = {}
    for db in JACKHMMER_DATABASES:
        known[f"{db}.fasta.gz"] = "Protein"
    for db in RNA_DATABASES:
        known[f"{db}.fasta.gz"] = "DNA/RNA"
    for db in HHBLITS_DATABASES + [BFD_DATABASE, CFDB_DATABASE]:
        known[f"{db}.tar.gz"] = "Protein"
    return known


def format_size(size_bytes: float) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def download_from_s3(*, bucket: str, key: str, destination: str) -> None:
    """Download a file from S3 using aws cli with progress bar and resume support."""
    s3_uri = f"s3://{bucket}/{key}"
    cmd = ["aws", "s3", "cp", "--no-sign-request", s3_uri, destination]
    logger.info(f"Downloading {s3_uri} to {destination}")
    sp.run(cmd, check=True)


def list_databases() -> None:
    """List all objects in the S3 bucket with known database indicators."""
    cmd = [
        "aws",
        "s3api",
        "list-objects-v2",
        "--no-sign-request",
        "--bucket",
        S3_BUCKET,
        "--prefix",
        S3_PREFIX + "/",
        "--output",
        "json",
    ]
    result = sp.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    known_dbs = get_known_database_info()

    rows = []
    for obj in data.get("Contents", []):
        key = obj["Key"]
        filename = key.removeprefix(S3_PREFIX + "/")
        if not filename:  # Skip the prefix itself
            continue
        size_bytes = obj["Size"]
        db_type = known_dbs.get(filename, "")
        known = "✓" if filename in known_dbs else ""
        rows.append((filename, format_size(size_bytes), db_type, known))

    # Print table
    if not rows:
        print("No objects found in bucket.")
        return

    # Calculate column widths
    headers = ("Filename", "Size", "Type", "Known")
    col_widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))
    ]

    # Print header
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, col_widths)))


def parse_args() -> Namespace:
    parser = ArgumentParser(description="OpenFold3 database downloader")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'list' subcommand
    subparsers.add_parser("list", help="List available databases in S3 bucket")

    # 'download' subcommand
    download_parser = subparsers.add_parser("download", help="Download databases")
    download_parser.add_argument("--output-dir", type=str, default="./alignment_dbs")
    download_parser.add_argument("--download-bfd", action="store_true")
    download_parser.add_argument("--download-cfdb", action="store_true")
    download_parser.add_argument("--download-rna-dbs", action="store_true")
    download_parser.add_argument(
        "--jackhmmer-dbs",
        type=str,
        nargs="+",
        default=None,
        help="Jackhmmer databases to download. "
        f"Defaults to all: {JACKHMMER_DATABASES}. "
        "Use 'uniref90 pdb_seqres' for a minimal test set.",
    )
    download_parser.add_argument(
        "--hhblits-dbs",
        type=str,
        nargs="+",
        default=None,
        help=f"HHblits databases to download. Defaults to: {HHBLITS_DATABASES}. "
        "Use --download-bfd and --download-cfdb to add those databases.",
    )

    return parser.parse_args()


def download(args: Namespace) -> None:
    base_outdir = Path(args.output_dir)
    base_outdir.mkdir(exist_ok=True, parents=True)

    # Build list of jackhmmer databases
    if args.jackhmmer_dbs is not None:
        jackhmmer_dbs = list(args.jackhmmer_dbs)
    else:
        jackhmmer_dbs = list(JACKHMMER_DATABASES)
    logger.info(f"Jackhmmer databases to process: {jackhmmer_dbs}")

    if args.download_rna_dbs:
        jackhmmer_dbs += list(RNA_DATABASES)
        logger.info(f"Including RNA databases: {RNA_DATABASES}")

    # Download jackhmmer databases
    for db in jackhmmer_dbs:
        output_filename = f"{base_outdir}/{db}/{db}.fasta.gz"
        if Path(output_filename).with_suffix("").exists():
            logger.info(f"{db} exists, skipping")
            continue
        outpath_db = Path(f"{base_outdir}/{db}/")
        outpath_db.mkdir(exist_ok=True, parents=True)
        logger.info(f"Downloading {db}...")
        download_from_s3(
            bucket=S3_BUCKET,
            key=f"{S3_PREFIX}/{db}.fasta.gz",
            destination=output_filename,
        )
        logger.info(f"Unzipping {db}...")
        sp.run(["gunzip", output_filename], check=True)

    # Build list of hhblits databases
    if args.hhblits_dbs is not None:
        hhblits_dbs = list(args.hhblits_dbs)
    else:
        hhblits_dbs = list(HHBLITS_DATABASES)
        if args.download_bfd:
            hhblits_dbs.append(BFD_DATABASE)
        if args.download_cfdb:
            hhblits_dbs.append(CFDB_DATABASE)
    logger.info(f"HHblits databases to process: {hhblits_dbs}")

    # Download hhblits databases
    for db in hhblits_dbs:
        output_filename = f"{base_outdir}/{db}/{db}.tar.gz"
        if Path(output_filename).parent.exists():
            logger.info(f"{db} exists, skipping")
            continue
        outpath_db = Path(f"{base_outdir}/{db}/")
        outpath_db.mkdir(exist_ok=True, parents=True)
        logger.info(f"Downloading {db}...")
        download_from_s3(
            bucket=S3_BUCKET,
            key=f"{S3_PREFIX}/{db}.tar.gz",
            destination=output_filename,
        )
        logger.info(f"Extracting {db}...")
        sp.run(
            ["tar", "xzf", output_filename, "-C", str(outpath_db.parent)],
            check=True,
        )
        # tar does not clean up, so manually delete
        Path(output_filename).unlink()


def main() -> None:
    args = parse_args()
    if args.command == "list":
        list_databases()
    elif args.command == "download":
        download(args)


if __name__ == "__main__":
    main()
