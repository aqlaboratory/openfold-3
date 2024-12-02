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
from openfold3.core.data.primitives.quality_control.asserts import ENSEMBLED_ASSERTS
from openfold3.core.data.primitives.quality_control.logging_utils import (
    F_NAME_ORDER,
    PDB_ID,
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
        """
        The following attributes are set in the worker_init_function_with_logging
        on a per-worker basis:
         - logger
         - compliance_log
         - processed_datapoint_log
         - runtime_token
         - mem_token
         - mem_log_token
         - mem_func_token
        """

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        """Returns a single datapoint from the dataset."""

        # Get PDB ID from the datapoint cache and the preferred chain/interface
        datapoint = self.datapoint_cache.iloc[index]
        pdb_id = datapoint["pdb_id"]
        PDB_ID.set(pdb_id)
        preferred_chain_or_interface = datapoint["datapoint"]
        sample_data = {}

        # Check if datapoint needs to be skipped
        if self.skip_datapoint(pdb_id, preferred_chain_or_interface):
            return {}

        self.logger.info(
            f"Processing datapoint {index}, PDB ID: {pdb_id}, preferred "
            f"chain/interface: {preferred_chain_or_interface}"
        )

        try:
            sample_data = self.create_all_features(
                pdb_id=pdb_id,
                preferred_chain_or_interface=preferred_chain_or_interface,
                return_atom_arrays=True,
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
                    sample_data["features"],
                    sample_data["atom_array_cropped"],
                    sample_data["atom_array"],
                    runtimes,
                )

            # Add PDB and chain/interface IDs to the memory log
            if self.log_memory:
                with open(
                    self.get_worker_path(subdirs=None, fname="memory_profile.log"), "a"
                ) as f:
                    chain_interface_str = self.stringify_chain_interface(
                        preferred_chain_or_interface
                    )
                    f.write(
                        f"pdb_id: {pdb_id}\npreferred_chain_or_interface: "
                        f"{chain_interface_str}\n\n\n"
                    )

            # Save features and/or atom array
            if (self.save_features == "per_datapoint") | (
                self.save_atom_array == "per_datapoint"
            ):
                self.save_features_atom_array(
                    sample_data["features"],
                    sample_data["atom_array_cropped"],
                    pdb_id,
                    preferred_chain_or_interface,
                )

            # Asserts
            if self.run_asserts:
                self.assert_full_compliance(
                    index,
                    sample_data["atom_array_cropped"],
                    pdb_id,
                    preferred_chain_or_interface,
                    sample_data["features"],
                    self.token_budget,
                    self.template.n_templates,
                )

            return sample_data["features"]

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
            ) & all([i in sample_data for i in ["features", "atom_array_cropped"]]):
                self.save_features_atom_array(
                    sample_data["features"],
                    sample_data["atom_array_cropped"],
                    pdb_id,
                    preferred_chain_or_interface,
                )
            if self.save_full_traceback:
                self.save_full_traceback_for_sample(
                    e, pdb_id, preferred_chain_or_interface
                )
            return {}

        finally:
            pass
            # Cannot actually do the following because it might happen that the final
            # datapoint finished being processed before previous datapoints finish being
            # processed do to the asynchronous nature of the workers ---
            # # Reset context variables before the worker shuts down
            # if index == len(self.__len__()) - 1:
            #     LOG_RUNTIMES.reset(self.runtime_token)
            #     LOG_MEMORY.reset(self.mem_token)
            #     WORKER_MEM_LOG_PATH.reset(self.mem_log_token)

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
            compliance_file = self.get_worker_path(subdirs=None, fname="passed_ids.tsv")
            self.compliance_log.save_worker_compliance_file(compliance_file)

    @staticmethod
    def stringify_chain_interface(preferred_chain_or_interface: str | list[str]) -> str:
        return (
            "-".join(preferred_chain_or_interface)
            if isinstance(preferred_chain_or_interface, list)
            else preferred_chain_or_interface
        )

    def save_features_atom_array(
        self, features, atom_array_cropped, pdb_id, preferred_chain_or_interface
    ):
        """Saves features and/or atom array from the worker process to disk."""
        chain_interface_str = self.stringify_chain_interface(
            preferred_chain_or_interface
        )
        log_output_feat = self.get_worker_path(
            subdirs=[pdb_id], fname=f"{chain_interface_str}_features.pkl"
        )
        log_output_aa = self.get_worker_path(
            subdirs=[pdb_id], fname=f"{chain_interface_str}_atom_array.pkl"
        )

        if self.save_features is not False:
            torch.save(
                features,
                log_output_feat,
            )
        if (self.save_atom_array is not False) & (atom_array_cropped is not None):
            with open(
                log_output_aa,
                "wb",
            ) as f:
                pkl.dump(atom_array_cropped, f)

    def save_full_traceback_for_sample(self, e, pdb_id, preferred_chain_or_interface):
        """Saves the full traceback to for failed samples."""
        chain_interface_str = self.stringify_chain_interface(
            preferred_chain_or_interface
        )
        log_output_errfile = self.get_worker_path(
            subdirs=[pdb_id], fname=f"{pdb_id}-{chain_interface_str}_error.log"
        )

        # Create temporary logger to log the traceback
        # This is necessary because we want to not save the traceback to the main logger
        # output file but to a pdb-entry specific directory
        sample_logger = logging.getLogger(f"{pdb_id}-{chain_interface_str}")
        if sample_logger.hasHandlers():
            sample_logger.handlers.clear()
        sample_logger.setLevel(self.logger.logger.level)
        sample_logger.propagate = False
        sample_file_handler = logging.FileHandler(
            log_output_errfile,
            mode="w",
        )
        sample_file_handler.setLevel(self.logger.logger.level)
        sample_logger.addHandler(sample_file_handler)

        sample_logger.error(
            f"Failed to process entry {pdb_id} chain/interface "
            f"{chain_interface_str}"
            f"\n\nException:\n{str(e)}"
            f"\n\nType:\n{type(e).__name__}"
            f"\n\nTraceback:\n{traceback.format_exc()}"
        )

        # Remove logger
        for h in sample_logger.handlers[:]:
            sample_logger.removeHandler(h)
            h.close()
        sample_logger.setLevel(logging.CRITICAL + 1)
        del logging.Logger.manager.loggerDict[f"{pdb_id}-{chain_interface_str}"]

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
            chain_interface_str = self.stringify_chain_interface(
                preferred_chain_or_interface
            )
            log_output_datafile = self.get_worker_path(
                subdirs=None, fname="datapoint_statistics.tsv"
            )

            # Init line:
            line = f"{pdb_id}\t{chain_interface_str}\t"

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

            # # radius of gyration
            # for aa in full_aa:
            #     if len(aa) > 0:
            #         aa_resolved = aa[~np.isnan(aa.coord).any(axis=1)]
            #         statistics += [struc.gyration_radius(aa_resolved)]
            #     else:
            #         statistics += ["NaN"]

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
                    statistics += [get_interface_string(aa_a, aa_b, "NaN")]
                else:
                    statistics += ["NaN"]

            # sub-pipeline runtimes
            statistics += list(runtimes)

            # Collate into tab format
            line += "\t".join(map(str, statistics))
            line += "\n"

            with open(log_output_datafile, "a") as f:
                f.write(line)

    def get_worker_path(self, subdirs: list[str] | None, fname: str | None) -> Path:
        """Returns the path to the worker output directory or file.

        Args:
            subdirs (list[str] | None):
                List of subdirectories to append to the worker output directory.
            fname (str | None):
                Filename to append to the worker output directory.

        Returns:
            Path:
                Path to the worker output directory or file. Without subdirs and fname
                this is log_output_directory/worker_{worker_id}.
        """
        log_output_path = self.logger.extra["log_output_directory"] / Path(
            "worker_{}".format(self.logger.extra["worker_id"])
        )
        if subdirs is not None:
            log_output_path = log_output_path / Path(*subdirs)
        log_output_path.mkdir(parents=True, exist_ok=True)
        if fname is not None:
            log_output_path = log_output_path / Path(fname)
        return log_output_path

    def fetch_runtimes(
        self,
    ) -> np.ndarray[float]:
        """Fetches sub-pipeline runtimes.

        Runtimes are collected into a single runtime dict of the topmost level function
        called directly in the getitem, which, by default, is create_all_features. To
        log the runtime of a specific function called in the getitem, make sure that
        1. it is decorated with the @log_runtime_memory decorator
        2. all higher-level functions in which it is called are also decorated with
        @log_runtime_memory
        3. its key str is in the F_NAME_ORDER list in logging_utils

        Args:
            top_function_call (list[callable]):
                A list of top-level sub-pipeline functions called in the getitem.

        Returns:
            np.ndarray[float]:
                Float of runtimes for each sub-pipeline.
        """
        # Create flat runtime dictionary
        # Sub-function runtimes are collected in the runtime attribute of the top-level
        # function wrapper directly
        runtime_dict = {}
        runtime_dict.update(self.create_all_features.runtime)

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
