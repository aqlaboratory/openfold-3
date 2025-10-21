import subprocess as sp
from argparse import ArgumentParser
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

parser = ArgumentParser()
parser.add_argument("--output-dir", type=str, default="./alignment_dbs")
parser.add_argument("--download-bfd", action="store_true")
parser.add_argument("--download-cfdb", action="store_true")
parser.add_argument("--download-rna-dbs", action="store_true")


def main(args):
    base_outdir = Path(args.output_dir)
    base_outdir.mkdir(exist_ok=True, parents=True)
    # Create an anonymous S3 client
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "openfold"
    pfx = "alignment_databases"
    ## download jackhmmer databases
    databases = ["uniprot", "uniref90", "mgnify", "pdb_seqres"]
    if args.download_rna_dbs:
        databases += ["rfam", "rnacentral", "nucleotide_collection"]
    for db in databases:
        output_filename_zip = f"{base_outdir}/{db}/{db}.fasta.gz"
        if Path(output_filename_zip).with_suffix("").exists():
            print(f"{db} exists, skipping")
            continue
        outpath_db = Path(f"{base_outdir}/{db}/")
        outpath_db.mkdir()
        print(f"Downloading {db}...")
        s3.download_file(bucket_name, f"{pfx}/{db}.fasta.gz", output_filename_zip)
        print(f"Unzipping {db}...")
        sp.run(["gunzip", output_filename_zip], check=True)

    ## download hhblits databases
    databases = ["uniref30"]
    if args.download_bfd:
        databases.append("bfd")
    if args.download_cfdb:
        databases.append("cfdb")

    for db in databases:
        output_filename_zip = f"{base_outdir}/{db}/{db}.tar.gz"
        if Path(output_filename_zip).parent.exists():
            print(f"{db} exists, skipping")
            continue
        outpath_db = Path(f"{base_outdir}/{db}/")
        outpath_db.mkdir()
        s3.download_file(bucket_name, f"{pfx}/{db}.tar.gz", output_filename_zip)
        sp.run(
            ["tar", "xzf", output_filename_zip, "-C", str(outpath_db.parent)],
            check=True,
        )
        # tar does not clean up, so manually delete
        Path(output_filename_zip).unlink()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
