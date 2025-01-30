"""Modified Dataset classes to support auxiliary logging features.

Supported use cases:
    - quality control logging
    - data statistics logging
    - logging of runtime and memory usage

"""

import bisect
import logging
import pickle as pkl
import traceback
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import biotite.structure as struc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from typing_extensions import deprecated

from openfold3.core.data.framework.data_module import DatasetMode, MultiDatasetConfig
from openfold3.core.data.framework.single_datasets.abstract_single import (
    DATASET_REGISTRY,
)
from openfold3.core.data.primitives.quality_control.asserts import ENSEMBLED_ASSERTS
from openfold3.core.data.primitives.quality_control.logging_utils import (
    F_NAME_ORDER,
    PDB_ID,
    RUNTIME_DICT,
    get_interface_string,
)
from openfold3.core.data.resources.residues import (
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    STANDARD_RNA_RESIDUES,
    MoleculeType,
)


def add_logging_to_dataset(dataset_cls):
    class LoggedDataset(LoggingMixin, dataset_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.wrapped_dataset_class = dataset_cls.__name__

    return LoggedDataset


class LoggingMixin:
    def __init__(
        self,
        run_asserts=None,
        save_features=None,
        save_atom_array=None,
        save_full_traceback=None,
        save_statistics=None,
        log_runtimes=None,
        log_memory=None,
        subset_to_examples=None,
        no_preferred_chain_or_interface=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.run_asserts = run_asserts
        self.save_features = save_features
        self.save_atom_array = save_atom_array
        self.save_full_traceback = save_full_traceback
        self.save_statistics = save_statistics
        self.log_runtimes = log_runtimes
        self.log_memory = log_memory
        if subset_to_examples is not None and len(subset_to_examples) > 0:
            self.subset_examples(subset_to_examples)

        if no_preferred_chain_or_interface:
            self.remove_preferred_chain_or_interface()
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
        The following attributes are set on a per-getitem-call basis inside the getitem
        of the ConcatDataset or StochasticSamplerDataset:
         - datapoint_idx
        """

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        """Returns a single datapoint from the dataset."""

        # Get PDB ID from the datapoint cache and the preferred chain/interface
        datapoint = self.datapoint_cache.iloc[index]
        pdb_id = datapoint["pdb_id"]
        preferred_chain_or_interface = datapoint.get(
            "preferred_chain_or_interface", None
        )
        datapoint_probability = datapoint.get("datapoint_probabilities", None)
        n_clust = datapoint.get("n_clust", None)

        # Set context variables and containers
        PDB_ID.set(pdb_id)
        RUNTIME_DICT.set({})
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
                return_crop_strategy=True,
            )

            # Fetch recorded runtimes
            if self.log_runtimes:
                runtimes = self.fetch_runtimes()
            else:
                runtimes = np.array([])

            # Save extra data
            if self.save_statistics:
                self.save_data_statistics(
                    index,
                    pdb_id,
                    preferred_chain_or_interface,
                    datapoint_probability,
                    n_clust,
                    sample_data,
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
                    self.n_tokens,
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
        preferred_chain_or_interface = (
            "nan"
            if preferred_chain_or_interface is None
            else preferred_chain_or_interface
        )
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
        index: int,
        pdb_id: str,
        preferred_chain_or_interface: str | list[str],
        datapoint_probability: float | None,
        n_clust: int | None,
        sample_data: dict[str, Any],
        runtimes: np.ndarray[str],
    ):
        """Saves additional data statistics.

        !!! IMPORTANT NOTE: If the collection and order of items logged in this function
        are changed, the worker_config > configure_extra_data_file > all_headers list
        must be updated accordingly, EXCEPT FOR RUNTIMES, which should be updated in the
        logging_utils > F_NAME_ORDER variable !!!
        """
        # TODO: add logic to jointly update the all_headers list in the logging_datasets
        # and the save_data_statistics method in the WeightedPDBDatasetWithLogging class
        if self.save_statistics:
            # Unpack sample_data
            features = sample_data["features"]
            atom_array_cropped = sample_data["atom_array_cropped"]
            atom_array = sample_data["atom_array"]
            crop_strategy = sample_data["crop_strategy"]

            # Set worker output directory
            chain_interface_str = self.stringify_chain_interface(
                preferred_chain_or_interface
            )
            log_output_datafile = self.get_worker_path(
                subdirs=None, fname="datapoint_statistics.tsv"
            )

            # Init line:
            line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t".format(
                self.logger.extra["worker_id"],
                self.datapoint_index,
                index,
                self.wrapped_dataset_class,
                pdb_id,
                chain_interface_str,
                datapoint_probability,
                n_clust,
                crop_strategy,
            )

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

            # TODO: improve gyration radius calculation speed by reducing IO overhead
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

            # Entry metadata from dataset cache
            statistics += [
                getattr(self.dataset_cache.structure_data[pdb_id], "resolution", None)
            ]
            statistics += [
                getattr(self.dataset_cache.structure_data[pdb_id], "release_date", None)
            ]

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
    ) -> np.ndarray[str]:
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
            np.ndarray[str]:
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

    def subset_examples(self, subset_to_examples: str) -> None:
        """Subsets the dataset_cache and datapoint_cache to a subset of examples.

        Args:
            subset_to_examples (str):
                Comma-separated list of PDB IDs to subset the dataset_cache and
                datapoint_cache to.
        """
        # Format input
        subset_to_examples = (
            subset_to_examples.split(",")
            if ((subset_to_examples is not None) or (len(subset_to_examples) > 0))
            else subset_to_examples
        )
        subset_to_examples = [ex.strip() for ex in subset_to_examples]

        # Subset dataset_cache
        structure_data = {
            ex: self.dataset_cache.structure_data[ex] for ex in subset_to_examples
        }
        self.dataset_cache.structure_data = structure_data

        # Subset datapoint_cache
        self.datapoint_cache = self.datapoint_cache[
            self.datapoint_cache["pdb_id"].isin(subset_to_examples)
        ]

    def remove_preferred_chain_or_interface(self) -> None:
        """Removes a preferred chain or interface from the datapoint_cache."""

        # Remove from datapoint_cache
        unique_pdb_ids = sorted(set(self.datapoint_cache["pdb_id"]))
        self.datapoint_cache = pd.DataFrame(
            {
                "pdb_id": unique_pdb_ids,
                "datapoint": [None] * len(unique_pdb_ids),
                "weight": [1] * len(unique_pdb_ids),
            }
        )


_T_co = TypeVar("_T_co", covariant=True)


class ConcatDataset(Dataset[_T_co]):
    """Dataset as a concatenation of multiple datasets.

    Taken from PyTorch's ConcatDataset implementation: https://github.com/pytorch/pytorch/blob/df458be4e5e96ce009ae1920da09a4095b34682e/torch/utils/data/dataset.py#L304
    for extendable class definition.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: list[Dataset[_T_co]]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), (
                "ConcatDataset does not support IterableDataset"
            )
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Set datapoint index in sampled dataset - used for logging in the treadmill
        self.datasets[dataset_idx].datapoint_idx = idx

        return self.datasets[dataset_idx][sample_idx]

    @property
    @deprecated(
        "`cummulative_sizes` attribute is renamed to `cumulative_sizes`",
        category=FutureWarning,
    )
    def cummulative_sizes(self):
        return self.cumulative_sizes

    def get_worker_path(self, subdirs: list[str] | None, fname: str) -> str:
        """Get the worker-specific path for logging memory.

        Note: this function only works if individual datasets passed to the
        StochasticSamplerDataset were wrapped with the LoggingMixin.

        Args:
            subdirs (list[str] | None):
                List of subdirectories to append to the path.
            fname (str):
                Filename to append to the path.
        """
        # Get the dataset-specific path
        dataset_path = self.datasets[0].get_worker_path(subdirs=subdirs, fname=fname)

        return dataset_path


def init_datasets_with_logging(
    multi_dataset_config: MultiDatasetConfig,
    type_to_init: DatasetMode,
    run_asserts: bool,
    save_features: bool,
    save_atom_array: bool,
    save_full_traceback: bool,
    save_statistics: bool,
    log_runtimes: bool,
    log_memory: bool,
    subset_to_examples: list[str],
    no_preferred_chain_or_interface: bool,
) -> Sequence[Dataset]:
    """Adds logging to the dataset classes and initializes them.

    Same as DataModule.init_datasets() with added logging support
    for each individual dataset.

    Args:
        multi_dataset_config (MultiDatasetConfig):
            Nested dataset config dicts.
        type_to_init (DatasetMode):
            Dataset mode to initialize.
        run_asserts (bool):
            Whether to run asserts.
        save_features (bool):
            Whether to save the featuredict upon error.
        save_atom_array (bool):
            Whether to save the atom array upon error.
        save_full_traceback (bool):
            Whether to save the full traceback upon error.
        save_statistics (bool):
            Whether to save additional data statistics.
        log_runtimes (bool):
            Whether to log runtimes.
        log_memory (bool):
            Whether to log memory usage.
        subset_to_examples (list[str]):
            List of PDB IDs to subset the dataset_cache and datapoint_cache to.
        no_preferred_chain_or_interface (bool):
            Whether to remove the preferred chain or interface from the datapoint_cache.

    Returns:
        Sequence[Dataset]:
            List of initialized datasets.
    """
    # Note that the dataset config already contains the paths!
    if type_to_init is None:
        types_to_init = [
            DatasetMode.train,
            DatasetMode.validation,
            DatasetMode.test,
            DatasetMode.prediction,
        ]
    else:
        types_to_init = [type_to_init]
    # TODO: add explicit for loop to make logic clearer
    datasets = [
        (
            add_logging_to_dataset(DATASET_REGISTRY[dataset_class])(
                run_asserts=run_asserts,
                save_features=save_features,
                save_atom_array=save_atom_array,
                save_full_traceback=save_full_traceback,
                save_statistics=save_statistics,
                log_runtimes=log_runtimes,
                log_memory=log_memory,
                subset_to_examples=subset_to_examples,
                no_preferred_chain_or_interface=no_preferred_chain_or_interface,
                dataset_config=dataset_config,
            )
        )
        for dataset_class, dataset_config, dataset_type in zip(
            multi_dataset_config.classes,
            multi_dataset_config.configs,
            multi_dataset_config.modes,
        )
        if dataset_type in types_to_init
    ]

    return datasets
