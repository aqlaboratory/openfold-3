"""Modified Dataset classes to support auxiliary logging features.

Supported use cases:
    - quality control logging
    - data statistics logging
    - logging of runtime and memory usage

"""

import logging
import pickle as pkl
import traceback
from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch

from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_ref_conformers_af3,
)
from openfold3.core.data.pipelines.featurization.loss_weights import set_loss_weights
from openfold3.core.data.pipelines.featurization.msa import featurize_msa_af3
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_af3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_templates_dummy_af3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    get_reference_conformer_data_af3,
)
from openfold3.core.data.pipelines.sample_processing.msa import process_msas_cropped_af3
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_structure_af3,
)
from openfold3.core.data.primitives.quality_control.asserts import ENSEMBLED_ASSERTS
from openfold3.core.data.primitives.quality_control.logging_utils import (
    F_NAME_ORDER,
    LOG_RUNTIMES,
    get_interface_string,
)
from openfold3.core.data.resources.residues import (
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    STANDARD_RNA_RESIDUES,
    MoleculeType,
)


class WeightedPDBDatasetWithLogging(WeightedPDBDataset):
    """Custom PDB dataset class with logging in the __getitem__."""

    def __init__(
        self,
        *args,
        run_asserts=None,
        save_features=None,
        save_atom_array=None,
        save_full_traceback=None,
        save_statistics=None,
        log_runtimes=None,
        log_memory=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.run_asserts = run_asserts
        self.save_features = save_features
        self.save_atom_array = save_atom_array
        self.save_full_traceback = save_full_traceback
        self.save_statistics = save_statistics
        self.log_runtimes = log_runtimes
        self.log_memory = log_memory
        # top-level pipelines for collecting runtimes
        self.top_f = [
            process_target_structure_af3,
            featurize_target_gt_structure_af3,
            process_msas_cropped_af3,
            featurize_msa_af3,
            featurize_templates_dummy_af3,
            get_reference_conformer_data_af3,
            featurize_ref_conformers_af3,
        ]
        """
        The following attributes are set in the worker_init_function_with_logging
        on a per-worker basis:
         - self.logger
         - self.compliance_log
         - self.processed_datapoint_log
        """

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        """Returns a single datapoint from the dataset."""
        # Get PDB ID from the datapoint cache and the preferred chain/interface
        datapoint = self.datapoint_cache.iloc[index]
        pdb_id = datapoint["pdb_id"]
        preferred_chain_or_interface = datapoint["datapoint"]
        features = {}
        atom_array_cropped = None

        # Set runtime logging context - should be thread-safe
        runtime_context_token = LOG_RUNTIMES.set(self.log_runtimes)

        # Check if datapoint needs to be skipped
        if self.skip_datapoint(pdb_id, preferred_chain_or_interface):
            return features

        self.logger.info(
            f"Processing datapoint {index}, PDB ID: {pdb_id}, preferred "
            f"chain/interface: {preferred_chain_or_interface}"
        )

        try:
            # Target structure and duplicate-expanded GT structure features
            atom_array_cropped, atom_array_gt, atom_array = (
                process_target_structure_af3(
                    target_structures_directory=self.target_structures_directory,
                    pdb_id=pdb_id,
                    crop_weights=self.crop_weights,
                    token_budget=self.token_budget,
                    preferred_chain_or_interface=preferred_chain_or_interface,
                    structure_format="pkl",
                    return_full_atom_array=True,
                )
            )
            # NOTE that for now we avoid the need for permutation alignment by providing
            # the cropped atom array as the ground truth atom array features.update(
            # featurize_target_gt_structure_af3( atom_array_cropped, atom_array_gt,
            #     self.token_budget ) )
            features.update(
                featurize_target_gt_structure_af3(
                    atom_array_cropped, atom_array_cropped, self.token_budget
                )
            )

            # MSA features
            msa_processed = process_msas_cropped_af3(
                alignments_directory=self.alignments_directory,
                alignment_db_directory=self.alignment_db_directory,
                alignment_index=self.alignment_index,
                atom_array=atom_array_cropped,
                data_cache_entry_chains=self.dataset_cache["structure_data"][pdb_id][
                    "chains"
                ],
                max_seq_counts={
                    "uniref90_hits": 10000,
                    "uniprot_hits": 50000,
                    "uniprot": 50000,
                    "bfd_uniclust_hits": 1000000,
                    "bfd_uniref_hits": 1000000,
                    "mgnify_hits": 5000,
                    "rfam_hits": 10000,
                    "rnacentral_hits": 10000,
                    "nucleotide_collection_hits": 10000,
                },
                token_budget=self.token_budget,
                max_rows_paired=8191,
            )
            features.update(featurize_msa_af3(msa_processed))

            # Dummy template features
            features.update(
                featurize_templates_dummy_af3(self.n_templates, self.token_budget)
            )

            # Reference conformer features
            processed_reference_molecules = get_reference_conformer_data_af3(
                atom_array=atom_array_cropped,
                per_chain_metadata=self.dataset_cache["structure_data"][pdb_id][
                    "chains"
                ],
                reference_mol_metadata=self.dataset_cache["reference_molecule_data"],
                reference_mol_dir=self.reference_molecule_directory,
            )
            features.update(featurize_ref_conformers_af3(processed_reference_molecules))

            # Loss switches
            features["loss_weights"] = set_loss_weights(
                self.loss_settings,
                self.dataset_cache["structure_data"][pdb_id]["resolution"],
            )

            # Fetch recorded runtimes
            if self.log_runtimes:
                runtimes = self.fetch_runtimes()
            else:
                runtimes = np.array([])

            # Save extra data
            if self.save_statistics:
                self.save_data_statistics(
                    pdb_id,
                    preferred_chain_or_interface,
                    features,
                    atom_array_cropped,
                    atom_array,
                    runtimes,
                )

            # Save features and/or atom array
            if (self.save_features == "per_datapoint") | (
                self.save_atom_array == "per_datapoint"
            ):
                self.save_features_atom_array(
                    features, atom_array_cropped, pdb_id, preferred_chain_or_interface
                )

            # Asserts
            if self.run_asserts:
                self.assert_full_compliance(
                    index,
                    atom_array_cropped,
                    pdb_id,
                    preferred_chain_or_interface,
                    features,
                    self.token_budget,
                    self.n_templates,
                )

            return features

        except Exception as e:
            # Catch all other errors
            self.logger.error(
                f"OTHER ERROR processing datapoint {index}, PDB ID: {pdb_id}"
            )
            self.logger.error(f"Error message: {e}")

            # Save features, atom array and per sample traceback
            if (
                (self.save_features == "on_error")
                | (self.save_atom_array == "on_error")
                | (self.save_features == "per_datapoint")
                | (self.save_atom_array == "per_datapoint")
            ):
                self.save_features_atom_array(
                    features, atom_array_cropped, pdb_id, preferred_chain_or_interface
                )
            if self.save_full_traceback:
                self.save_full_traceback_for_sample(
                    e, pdb_id, preferred_chain_or_interface
                )
            return features

        finally:
            # Reset context variable
            LOG_RUNTIMES.reset(runtime_context_token)

    def skip_datapoint(self, pdb_id, preferred_chain_or_interface):
        """Determines whether to skip a datapoint."""
        # Skip datapoint if it's in the compliance log and run_asserts is True or
        # if it's in the processed_datapoint_log and save_statistics is True
        if self.run_asserts | self.save_statistics:
            skip_datapoint = (
                f"{pdb_id}-{preferred_chain_or_interface}"
                in self.compliance_log.passed_ids
            ) | (f"{pdb_id}" in self.processed_datapoint_log)
        else:
            skip_datapoint = False
        return skip_datapoint

    def assert_full_compliance(
        self,
        index,
        atom_array_cropped,
        pdb_id,
        preferred_chain_or_interface,
        features,
        token_budget,
        n_templates,
    ):
        """Asserts that the getitem runs and all asserts pass."""
        # Get list of argument for the full list of asserts
        ensembled_args = [(features,)] * 17
        ensembled_args[2] = (features, token_budget)
        ensembled_args[12] = (features, token_budget, n_templates)
        # Get compliance array
        compliance = np.zeros(len(ENSEMBLED_ASSERTS))
        # Iterate over asserts and update compliance array
        try:
            for i, (assert_i, args_i) in enumerate(
                zip(ENSEMBLED_ASSERTS, ensembled_args)
            ):
                assert_i(*args_i)
                compliance[i] = 1
        except AssertionError as e:
            # Catch assertion errors
            self.logger.error(
                f"ASSERTION ERROR processing datapoint {index}, PDB ID: {pdb_id}"
            )
            self.logger.error(f"Error message: {e}")

            # Save features and atom array
            if (
                (self.save_features == "on_error")
                | (self.save_atom_array == "on_error")
                | (self.save_features == "per_datapoint")
                | (self.save_atom_array == "per_datapoint")
            ):
                self.save_features_atom_array(
                    features, atom_array_cropped, pdb_id, preferred_chain_or_interface
                )
            if self.save_full_traceback:
                self.save_full_traceback_for_sample(
                    e, pdb_id, preferred_chain_or_interface
                )

        # Add IDs to compliance log if all asserts pass
        if compliance.all():
            self.compliance_log.passed_ids.add(
                f"{pdb_id}-{preferred_chain_or_interface}"
            )
            log_output_dir = self.logger.extra["log_output_directory"] / Path(
                "worker_{}".format(self.logger.extra["worker_id"])
            )
            self.compliance_log.save_worker_compliance_file(
                log_output_dir / Path("passed_ids.tsv")
            )

    def save_features_atom_array(
        self, features, atom_array_cropped, pdb_id, preferred_chain_or_interface
    ):
        """Saves features and/or atom array from the worker process to disk."""
        log_output_dir = self.logger.extra["log_output_directory"] / Path(
            "worker_{}/{}".format(self.logger.extra["worker_id"], pdb_id)
        )
        log_output_dir.mkdir(parents=True, exist_ok=True)

        preferred_chain_or_interface = (
            "-".join(preferred_chain_or_interface)
            if isinstance(preferred_chain_or_interface, list)
            else preferred_chain_or_interface
        )
        if self.save_features is not False:
            torch.save(
                features,
                log_output_dir
                / Path(f"{pdb_id}-{preferred_chain_or_interface}_features.pt"),
            )
        if (self.save_atom_array is not False) & (atom_array_cropped is not None):
            with open(
                log_output_dir
                / Path(f"{pdb_id}-{preferred_chain_or_interface}_atom_array.pkl"),
                "wb",
            ) as f:
                pkl.dump(atom_array_cropped, f)

    def save_full_traceback_for_sample(self, e, pdb_id, preferred_chain_or_interface):
        """Saves the full traceback to for failed samples."""

        log_output_dir = self.logger.extra["log_output_directory"] / Path(
            "worker_{}/{}".format(self.logger.extra["worker_id"], pdb_id)
        )
        log_output_dir.mkdir(parents=True, exist_ok=True)

        preferred_chain_or_interface = (
            "-".join(preferred_chain_or_interface)
            if isinstance(preferred_chain_or_interface, list)
            else preferred_chain_or_interface
        )

        # Create temporary logger to log the traceback
        # This is necessary because we want to not save the traceback to the main logger
        # output file but to a pdb-entry specific directory
        sample_logger = logging.getLogger(f"{pdb_id}-{preferred_chain_or_interface}")
        if sample_logger.hasHandlers():
            sample_logger.handlers.clear()
        sample_logger.setLevel(self.logger.logger.level)
        sample_logger.propagate = False
        sample_file_handler = logging.FileHandler(
            log_output_dir / Path(f"{pdb_id}-{preferred_chain_or_interface}_error.log"),
            mode="w",
        )
        sample_file_handler.setLevel(self.logger.logger.level)
        sample_logger.addHandler(sample_file_handler)

        sample_logger.error(
            f"Failed to process entry {pdb_id} chain/interface "
            f"{preferred_chain_or_interface}"
            f"\n\nException:\n{str(e)}"
            f"\n\nType:\n{type(e).__name__}"
            f"\n\nTraceback:\n{traceback.format_exc()}"
        )

        # Remove logger
        for h in sample_logger.handlers[:]:
            sample_logger.removeHandler(h)
            h.close()
        sample_logger.setLevel(logging.CRITICAL + 1)
        del logging.Logger.manager.loggerDict[
            f"{pdb_id}-{preferred_chain_or_interface}"
        ]

    def save_data_statistics(
        self,
        pdb_id,
        preferred_chain_or_interface,
        features,
        atom_array_cropped,
        atom_array,
        runtimes,
    ):
        """Saves additional data statistics."""
        if self.save_statistics:
            # Set worker output directory
            log_output_dir = self.logger.extra["log_output_directory"] / Path(
                "worker_{}".format(self.logger.extra["worker_id"])
            )
            preferred_chain_or_interface = (
                "-".join(preferred_chain_or_interface)
                if isinstance(preferred_chain_or_interface, list)
                else preferred_chain_or_interface
            )

            # Init line:
            line = f"{pdb_id}\t{preferred_chain_or_interface}\t"

            # Get per-molecule type atom arrays/residue starts
            atom_array_protein = atom_array[
                atom_array.molecule_type_id == MoleculeType.PROTEIN
            ]
            atom_array_protein_cropped = atom_array_cropped[
                atom_array_cropped.molecule_type_id == MoleculeType.PROTEIN
            ]
            atom_array_rna = atom_array[atom_array.molecule_type_id == MoleculeType.RNA]
            atom_array_rna_cropped = atom_array_cropped[
                atom_array_cropped.molecule_type_id == MoleculeType.RNA
            ]
            atom_array_dna = atom_array[atom_array.molecule_type_id == MoleculeType.DNA]
            atom_array_dna_cropped = atom_array_cropped[
                atom_array_cropped.molecule_type_id == MoleculeType.DNA
            ]
            atom_array_ligand = atom_array[
                atom_array.molecule_type_id == MoleculeType.LIGAND
            ]
            atom_array_ligand_cropped = atom_array_cropped[
                atom_array_cropped.molecule_type_id == MoleculeType.LIGAND
            ]
            residue_starts = struc.get_residue_starts(atom_array)
            residue_starts = (
                np.append(residue_starts, -1)
                if residue_starts[-1] != len(atom_array)
                else residue_starts
            )
            residue_starts_cropped = struc.get_residue_starts(atom_array_cropped)
            residue_starts_cropped = (
                np.append(residue_starts_cropped, -1)
                if residue_starts_cropped[-1] != len(atom_array_cropped)
                else residue_starts_cropped
            )

            # Get atom array lists for easier iteration
            all_aa = [
                atom_array,
                atom_array_cropped,
                atom_array_protein,
                atom_array_protein_cropped,
                atom_array_rna,
                atom_array_rna_cropped,
                atom_array_dna,
                atom_array_dna_cropped,
                atom_array_ligand,
                atom_array_ligand_cropped,
            ]
            full_aa = [
                atom_array,
                atom_array_cropped,
            ]
            per_moltype_aa = [
                atom_array_protein,
                atom_array_protein_cropped,
                atom_array_rna,
                atom_array_rna_cropped,
                atom_array_dna,
                atom_array_dna_cropped,
                atom_array_ligand,
                atom_array_ligand_cropped,
            ]
            polymer_aa = [
                atom_array_protein,
                atom_array_protein_cropped,
                atom_array_rna,
                atom_array_rna_cropped,
                atom_array_dna,
                atom_array_dna_cropped,
            ]

            # Collect data
            statistics = []

            # Number of atoms:
            for aa in all_aa:
                statistics += [len(aa)]

            # Number of residues
            for aa in polymer_aa:
                resid_tensor = torch.tensor(aa.res_id)
                statistics += [len(torch.unique_consecutive(resid_tensor))]

            # Unresolved data
            for aa, rs in zip(full_aa, [residue_starts, residue_starts_cropped]):
                if len(aa) > 0:
                    # Number of unresolved atoms
                    statistics += [np.isnan(aa.coord).any(axis=1).sum()]
                    # Number of unresolved residues
                    cumsums = np.cumsum(np.isnan(aa.coord).any(axis=1))
                    statistics += [(np.diff(cumsums[rs]) > 0).sum()]
                else:
                    statistics += ["NaN", "NaN"]

            # Number of chains
            for aa in full_aa:
                statistics += [len(set(aa.chain_id))]

            # Number of entities
            for aa in per_moltype_aa:
                statistics += [len(set(aa.entity_id))]

            # MSA depth
            msa = features["msa"]
            statistics += [msa.shape[0]]
            # Number of paired MSA rows
            statistics += [features["num_paired_seqs"].item()]
            # number of tokens with any aligned MSA columns in the crop
            statistics += [(msa.sum(dim=0)[:, -1] < msa.size(0)).sum().item()]

            # number of templates
            tbfm = features["template_backbone_frame_mask"]
            statistics += [(tbfm == 1).any(dim=-1).sum().item()]
            # number of tokens with any aligned template columns in the crop
            statistics += [(tbfm == 1).any(dim=-2).sum().item()]

            # number of tokens
            for aa in full_aa:
                statistics += [len(set(aa.token_id))]

            # Atomized residue token data
            for aa, vocab in zip(
                polymer_aa,
                [STANDARD_PROTEIN_RESIDUES_3] * 2
                + [STANDARD_RNA_RESIDUES] * 2
                + [STANDARD_DNA_RESIDUES] * 2,
            ):
                if len(aa) > 0:
                    # number of residue tokens atomized due to special
                    is_special_aa = ~np.isin(aa.res_name, vocab)
                    rs = struc.get_residue_starts(aa)
                    statistics += [is_special_aa[rs].sum()]

                    # number of residue tokens atomized due to covalent modifications
                    is_standard_atomized_aa = (~is_special_aa) & aa.is_atomized
                    statistics += [is_standard_atomized_aa[rs].sum()]
                else:
                    statistics += ["NaN", "NaN"]

            # radius of gyration
            for aa in full_aa:
                if len(aa) > 0:
                    aa_resolved = aa[~np.isnan(aa.coord).any(axis=1)]
                    statistics += [struc.gyration_radius(aa_resolved)]
                else:
                    statistics += ["NaN"]

            # interface statistics
            for aa_a, aa_b in zip(
                [
                    atom_array_protein,
                    atom_array_protein_cropped,
                ]
                * 4,
                per_moltype_aa,
            ):
                if (len(aa_a) > 0) & (len(aa_b) > 0):
                    statistics += [get_interface_string(aa_a, aa_b)]
                else:
                    statistics += ["NaN"]

            # sub-pipeline runtimes
            statistics += list(runtimes)

            # Collate into tab format
            line += "\t".join(map(str, statistics))
            line += "\n"

            with open(
                log_output_dir / Path("datapoint_statistics.tsv"),
                "a",
            ) as f:
                f.write(line)

    def fetch_runtimes(
        self,
    ) -> np.ndarray[float]:
        """Fetches sub-pipeline runtimes.

        Each sub-pipeline variable is decorated with a wrapper which stores its
        runtime and that of any child functions.

        Args:
            top_f (list[callable]):
                A list of top-level sub-pipeline functions called in the getitem.

        Returns:
            np.ndarray[float]:
                Float of runtimes for each sub-pipeline.
        """
        # Create flat runtime dictionary
        # Sub-function runtimes are collected in the runtime attribute of the top-level
        # function wrapper directly
        runtime_dict = {}
        for f in self.top_f:
            runtime_dict.update(f.runtime)

        # Get runtimes in order - 0 for any non-called function
        runtimes = np.array([runtime_dict.get(n, 0.0) for n in F_NAME_ORDER])

        # Log runtimes
        if not self.save_statistics:
            self.logger.info(
                "Rutimes:\n"
                + "\t".join(F_NAME_ORDER)
                + "\n"
                + "\t".join(map(str, runtimes))
            )

        return runtimes
