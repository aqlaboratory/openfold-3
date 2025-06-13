"""MSA pre-parsing script for AF3 dataset."""

import json
import multiprocessing as mp
from functools import wraps
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from openfold3.core.data.io.sequence.msa import parse_msas_direct, standardize_filepaths


@click.command()
@click.option(
    "--alignments_directory",
    required=True,
    help="Directory containing per-chain folders with multiple sequence alignments.",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--alignment_array_directory",
    required=True,
    help="Output directory to which the per-chain MSA npz files are to be saved.",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--max_seq_counts",
    required=True,
    type=str,
    help="The argument max_seq_counts "
    "input as a JSON string. Key must match the msa filenames without "
    "extension that are in the per-chain alignment directories. Alignments "
    "whose names do not match any key in max_seq_counts will not be parsed. "
    "Values are the maximum number of sequences to parse from the alignment.",
)
@click.option(
    "--num_workers",
    required=True,
    type=int,
    help=(
        "Number of workers to parallelize the template cache computation and filtering"
        " over."
    ),
)
def main(
    alignments_directory: Path,
    alignment_array_directory: Path,
    max_seq_counts: str,
    num_workers: int,
):
    """Preparse multiple sequence alignments for AF3 dataset."""
    try:
        max_seq_counts = json.loads(max_seq_counts)
    except json.JSONDecodeError:
        click.echo("Invalid max_seq_counts JSON string!")

    rep_chain_dir_iterator = [it.name for it in alignments_directory.iterdir()]

    # Create template cache for each query chain
    wrapped_msa_preparser = _MsaPreparser(
        alignments_directory, alignment_array_directory, max_seq_counts
    )
    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapped_msa_preparser,
                rep_chain_dir_iterator,
                chunksize=1,
            ),
            total=len(rep_chain_dir_iterator),
            desc="Pre-parsing MSAs",
        ):
            pass


def preparse_msas(
    alignments_directory: Path,
    alignment_array_directory: Path,
    max_seq_counts: dict[str, int],
    rep_pdb_chain_id: str,
) -> None:
    file_list = standardize_filepaths(alignments_directory / Path(rep_pdb_chain_id))
    msas = parse_msas_direct(
        file_list=file_list,
        max_seq_counts=max_seq_counts,
    )
    alignment_array_directory.mkdir(parents=True, exist_ok=True)

    msas_preparsed = {}
    for k, v in msas.items():
        msas_preparsed[k] = v.to_dict()

    np.savez_compressed(
        alignment_array_directory / Path(f"{rep_pdb_chain_id}.npz"), **msas_preparsed
    )


class _MsaPreparser:
    def __init__(
        self,
        alignments_directory: Path,
        alignment_array_directory: Path,
        max_seq_counts: dict[str, int],
    ) -> None:
        """Wrapper class for pre-parsing a directory of raw MSA files.

        This wrapper around `preparse_msas` is needed for multiprocessing, so that we
        can pass the constant arguments in a convenient way catch any errors that would
        crash the workers, and change the function call to accept a single Iterable.

        The wrapper is written as a class object because multiprocessing doesn't support
        decorator-like nested functions.

        Attributes:
            alignments_directory:
                Directory containing per-chain folders with multiple sequence
                alignments.
            alignment_array_directory:
                Output directory to which the per-chain MSA npz files are to be saved.

        """
        self.alignments_directory = alignments_directory
        self.alignment_array_directory = alignment_array_directory
        self.max_seq_counts = max_seq_counts

    @wraps(preparse_msas)
    def __call__(self, rep_pdb_chain_id: str) -> None:
        try:
            preparse_msas(
                self.alignments_directory,
                self.alignment_array_directory,
                self.max_seq_counts,
                rep_pdb_chain_id,
            )
        except Exception as e:
            print(f"Failed to preparse MSAs for chain {rep_pdb_chain_id}:\n{e}\n")


if __name__ == "__main__":
    main()
