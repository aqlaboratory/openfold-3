import json
import multiprocessing as mp
import traceback
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from openfold3.core.data.io.sequence.fasta import get_chain_id_to_seq_from_fasta
from openfold3.core.data.io.sequence.msa import parse_msas_preparsed
from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.primitives.structure.labels import get_id_starts
from openfold3.core.data.resources.residues import (
    DNA_RESTYPE_3TO1,
    MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1,
    PROTEIN_RESTYPE_3TO1,
    RNA_RESTYPE_3TO1,
    MoleculeType,
)

MOLTYPE_TO_3_TO_1_MAP = {
    MoleculeType.PROTEIN: PROTEIN_RESTYPE_3TO1,
    MoleculeType.DNA: DNA_RESTYPE_3TO1,
    MoleculeType.RNA: RNA_RESTYPE_3TO1,
}


def cross_validate_atomarray_msa(
    pdb_id: str,
    chain_to_rep: dict,
    target_structures_directory: Path,
    alignment_array_directory: Path,
) -> tuple[list, dict, dict]:
    mismatches = []
    mismatched_seqs_atom_array = {}
    mismatched_seqs_msa = {}

    atom_array = read_atomarray_from_npz(
        target_structures_directory / f"{pdb_id}/{pdb_id}.npz"
    )

    chain_ids = np.unique(atom_array.chain_id)

    for chain_id in chain_ids:
        try:
            rep_id = chain_to_rep.get(str(chain_id))

            if rep_id is None:
                continue

            atom_array_chain = atom_array[atom_array.chain_id == chain_id]

            mol_type_id_chain = np.unique(atom_array_chain.molecule_type_id)
            if len(mol_type_id_chain) != 1:
                raise ValueError(
                    f"Multiple molecule type IDs found in chain {str(chain_id)}: "
                    f"{mol_type_id_chain}"
                )
            mol_type_chain = MoleculeType(mol_type_id_chain[0])

            residue_starts = get_id_starts(atom_array_chain, "res_id")
            res_names_atom_array = atom_array_chain[residue_starts].res_name

            seq_structure = np.vectorize(
                MOLTYPE_TO_3_TO_1_MAP[mol_type_chain].get, otypes=[str]
            )(res_names_atom_array, MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1[mol_type_chain])

            chain_msa_data = parse_msas_preparsed(
                [alignment_array_directory / f"{rep_id}.npz"]
            )

            for k in chain_msa_data:
                seq_msa_array = chain_msa_data[k].msa[0, :]
                sm = "".join(seq_msa_array).replace("-", "").replace(".", "")
                sa = "".join(seq_structure)
                if sm != sa:
                    mismatches.append((pdb_id, str(chain_id), rep_id, k))
                    seq_key = (pdb_id, str(chain_id))
                    if seq_key not in mismatched_seqs_atom_array:
                        mismatched_seqs_atom_array[seq_key] = sa
                        mismatched_seqs_msa[seq_key] = sm

        except Exception as e:
            warnings.warn(
                f"Error processing {pdb_id} chain {chain_id}: {e}"
                "Traceback:\n" + str(traceback.format_exc()),
                stacklevel=2,
            )
            continue

    return mismatches, mismatched_seqs_atom_array, mismatched_seqs_msa


def cross_validate_fasta_msa(
    pdb_id: str,
    chain_to_rep: dict,
    target_structures_directory: Path,
    alignment_array_directory: Path,
) -> tuple[list, dict, dict]:
    mismatches = []
    mismatched_seqs_atom_array = {}
    mismatched_seqs_msa = {}

    chain_id_to_seq = get_chain_id_to_seq_from_fasta(
        target_structures_directory / f"{pdb_id}/{pdb_id}.fasta"
    )

    for chain_id, seq_structure in chain_id_to_seq.items():
        rep_id = chain_to_rep.get(str(chain_id))
        if rep_id is None:
            print(f"Warning: No representative ID found for {pdb_id} chain {chain_id}")
            continue
        chain_msa_data = parse_msas_preparsed(
            [alignment_array_directory / f"{rep_id}.npz"]
        )
        for k in chain_msa_data:
            seq_msa_array = chain_msa_data[k].msa[0, :]
            sm = "".join(seq_msa_array).replace("-", "").replace(".", "")
            sa = "".join(seq_structure)
            if sm != sa:
                mismatches.append((pdb_id, str(chain_id), rep_id, k))
                seq_key = (pdb_id, str(chain_id))
                if seq_key not in mismatched_seqs_atom_array:
                    mismatched_seqs_atom_array[seq_key] = sa
                    mismatched_seqs_msa[seq_key] = sm

    return mismatches, mismatched_seqs_atom_array, mismatched_seqs_msa


class AtomArrayMsaCrossValidator:
    def __init__(
        self,
        target_structures_directory: Path,
        structure_file_format: str,
        alignment_array_directory: Path,
        dataset_cache_file: Path,
        output_directory: Path,
        n_processes: int,
        chunksize: int,
    ):
        self.target_structures_directory = target_structures_directory
        self.structure_file_format = structure_file_format
        self.alignment_array_directory = alignment_array_directory
        with open(dataset_cache_file) as f:
            self.dataset_cache = json.load(f)
        self.pdb_ids = list(self.dataset_cache["structure_data"].keys())
        self.output_directory = output_directory
        self.n_processes = n_processes
        self.chunksize = chunksize

    def __call__(self) -> None:
        mismatches, mismatched_seqs_atom_array, mismatched_seqs_msa = [], {}, {}

        with mp.Pool(self.n_processes) as pool:
            for (
                mismatches_i,
                mismatched_seqs_atom_array_i,
                mismatched_seqs_msa_i,
            ) in tqdm(
                pool.imap_unordered(
                    self.cross_validate_atomarray_msa_safe,
                    self.pdb_ids,
                    chunksize=self.chunksize,
                ),
                total=len(self.pdb_ids),
                desc="Cross-validating sequences between AtomArrays and MSAs",
            ):
                mismatches.extend(mismatches_i)
                mismatched_seqs_atom_array.update(mismatched_seqs_atom_array_i)
                mismatched_seqs_msa.update(mismatched_seqs_msa_i)

        # Save to tsv
        print("Saving results metadata")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        mismatches_df = pd.DataFrame(
            mismatches, columns=["pdb_id", "chain_id", "rep_id", "database_name"]
        )
        mismatches_df.to_csv(
            self.output_directory / "mismatches.tsv", sep="\t", index=False
        )
        print(f"Total mismatches: {len(mismatches)}")
        # Save to fasta
        print("Saving mismatched sequences from AtomArrays to fasta")
        with open(self.output_directory / "mismatched_seqs_atom_array.fasta", "w") as f:
            for (pdb_id, chain_id), seq in tqdm(
                mismatched_seqs_atom_array.items(),
                desc="Writing fasta",
                total=len(mismatched_seqs_atom_array),
            ):
                f.write(f">{pdb_id}_{chain_id}\n{seq}\n")
        print("Saving mismatched sequences from MSAs to fasta")
        with open(self.output_directory / "mismatched_seqs_msa.fasta", "w") as f:
            for (pdb_id, chain_id), seq in tqdm(
                mismatched_seqs_msa.items(),
                desc="Writing fasta",
                total=len(mismatched_seqs_msa),
            ):
                f.write(f">{pdb_id}_{chain_id}\n{seq}\n")

    def cross_validate_atomarray_msa_safe(
        self,
        pdb_id: str,
    ) -> tuple[list, dict, dict]:
        try:
            chain_to_rep = {
                k: v["alignment_representative_id"]
                for k, v in self.dataset_cache["structure_data"][pdb_id][
                    "chains"
                ].items()
                if v["alignment_representative_id"] is not None
            }
            if self.structure_file_format == "npz":
                return cross_validate_atomarray_msa(
                    pdb_id,
                    chain_to_rep,
                    self.target_structures_directory,
                    self.alignment_array_directory,
                )
            elif self.structure_file_format == "fasta":
                return cross_validate_fasta_msa(
                    pdb_id,
                    chain_to_rep,
                    self.target_structures_directory,
                    self.alignment_array_directory,
                )
            else:
                raise ValueError(
                    f"Unsupported structure file format: {self.structure_file_format}"
                    ". Supported formats are 'npz' and 'fasta'."
                )
        except Exception as e:
            error_msg = f"Error processing {pdb_id}: {e}\nTraceback:\n" + str(
                traceback.format_exc()
            )
            with open(self.output_directory / "log.txt", "a") as log_f:
                log_f.write(error_msg)
            print(error_msg)
            return [], {}, {}


@click.command()
@click.option(
    "--target_structures_directory",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--structure_file_format",
    required=True,
    type=str,
)
@click.option(
    "--alignment_array_directory",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--dataset_cache_file",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--output_directory",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--n_processes",
    required=True,
    type=int,
)
@click.option(
    "--chunksize",
    default=1,
    type=int,
)
def main(
    target_structures_directory: Path,
    structure_file_format: str,
    alignment_array_directory: Path,
    dataset_cache_file: Path,
    output_directory: Path,
    n_processes: int,
    chunksize: int,
):
    cross_validator = AtomArrayMsaCrossValidator(
        target_structures_directory,
        structure_file_format,
        alignment_array_directory,
        dataset_cache_file,
        output_directory,
        n_processes,
        chunksize,
    )
    cross_validator()


if __name__ == "__main__":
    main()
