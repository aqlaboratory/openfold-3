import logging
from pathlib import Path

import click
from tqdm import tqdm

from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
    write_multichain_fasta,
)
from openfold3.core.data.primitives.caches.format import PreprocessingDataCache
from openfold3.core.data.resources.residues import MoleculeType

logger = logging.getLogger("openfold3")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


@click.command()
@click.argument(
    "preprocessed_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "out_file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--uniquify",
    is_flag=True,
    default=False,
    help="If set, only retain one entry per unique sequence.",
)
def main(preprocessed_dir: Path, out_file: Path, uniquify: bool = False):
    """
    Collects individual preprocessed FASTA files into a single multi-chain FASTA file.

    The FASTA is written with the following format:

    >PDB_ID, renum_asym=CHAIN_ID, auth_asym=AUTH_ASYM_ID, label_asym=LABEL_ASYM_ID, mol_type=MOL_TYPE, date=RELEASE_DATE
    sequence

    PREPROCESSED_DIR: Path to the directory created by preprocess_pdb_af3.py
    OUT_FILE: Path to the output consolidated FASTA file.
    """  # noqa: E501
    metadata_cache_file = preprocessed_dir / "metadata.json"
    preprocessed_structure_dir = preprocessed_dir / "structure_files"

    # Get the structure data from the metadata cache
    logger.info("Reading metadata cache...")
    structure_cache = PreprocessingDataCache.from_json(
        metadata_cache_file
    ).structure_data

    # Get the mapping of {PDB_ID}_{chain_ID} to sequence for all preprocessed files
    id_to_seq = consolidate_preprocessed_fastas(preprocessed_structure_dir)

    # Only retain one ID per sequence
    if uniquify:
        logger.info("Uniquifying sequences...")
        seq_to_id = {seq: seq_id for seq_id, seq in id_to_seq.items()}
        id_to_seq = {seq_id: seq for seq, seq_id in seq_to_id.items()}

    # Create new mapping with headers to use in the final FASTA
    header_to_seq = {}

    for seq_id, seq in tqdm(id_to_seq.items(), desc="Creating new headers"):
        pdb_id, chain_id = seq_id.split("_")

        entry_data = structure_cache[pdb_id]
        chain_data = entry_data.chains[chain_id]

        molecule_type_str = MoleculeType(chain_data.molecule_type).name

        header = (
            f">{pdb_id}, renum_asym={chain_id}, auth_asym={chain_data.auth_asym_id}, "
            f"label_asym={chain_data.label_asym_id}, "
            f"mol_type={molecule_type_str}, date={entry_data.release_date}"
        )

        header_to_seq[header] = seq

    # Write the new FASTA file
    logger.info(f"Writing new FASTA file to {out_file}")

    write_multichain_fasta(out_file, header_to_seq, sort=True)
    logger.info("Done.")


if __name__ == "__main__":
    main()
