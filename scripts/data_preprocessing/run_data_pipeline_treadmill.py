"""
Script to iterate over datapoints with the datapipeline.

Components:
- WeightedPDBDatasetWithLogging:
    - Custom WeightedPDBDataset class that catches asserts and exceptions in the
    __getitem__.
    - also allows for saving features and atom array when an exception occurs
    in a worker process.
- worker_init_function_with_logging:
    Custom worker init function with per-worker logging and feature/atom array saving.
"""

import logging
import os
import pickle as pkl
import random
import sys
import traceback
from pathlib import Path

import biotite.structure as struc
import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from biotite.structure import AtomArray
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import _NUMPY_AVAILABLE
from openfold3.core.data.framework.lightning_utils import _generate_seed_sequence
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
from openfold3.core.data.primitives.structure.interface import (
    get_query_interface_atom_pair_idxs,
)
from openfold3.core.data.resources.residues import (
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    STANDARD_RNA_RESIDUES,
    MoleculeType,
)
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.projects import registry

np.set_printoptions(threshold=sys.maxsize)

# if importlib.util.find_spec("deepspeed") is not None:
#     import deepspeed

#     # TODO: Resolve this
#     # This is a hack to prevent deepspeed from doing the triton matmul autotuning
#     # I'm not sure why it's doing this by default, but it's causing the tests to hang
#     deepspeed.HAS_TRITON = False


# Asserts
def assert_no_nan_inf(features):
    """Checks if any tensor in the features dictionary contains NaNs or infs."""
    for key, entry in features.items():
        if isinstance(entry, dict):
            for subkey, subentry in entry.items():
                assert ~torch.isnan(subentry).any(), f"Tensor '{subkey}' contains NaNs."
                assert ~torch.isinf(subentry).any(), f"Tensor '{subkey}' contains infs."
        else:
            assert ~torch.isnan(entry).any(), f"Tensor '{key}' contains NaNs."
            assert ~torch.isinf(entry).any(), f"Tensor '{key}' contains infs."


def assert_token_atom_sum_match(features):
    """Checks if the sums of numbers of atoms in different tensors match."""

    # Match total sizes of
    sizes = []
    # - broadcast_token_feat_to_atoms
    sizes.append(
        broadcast_token_feat_to_atoms(
            token_mask=features["token_mask"],
            num_atoms_per_token=features["num_atoms_per_token"],
            token_feat=features["token_index"],
        ).shape[0]
    )
    # - ref_* features
    sizes.extend([f.shape[0] for k, f in features.items() if k.startswith("ref_")])
    # - GT N atoms
    sizes.extend(
        [
            f.shape[0]
            for k, f in features["ground_truth"].items()
            if k.startswith("atom_")
        ]
    )
    # - sum of num atoms per token
    sizes.append(features["num_atoms_per_token"].sum().item())

    assert len(set(sizes)) == 1, "Mismatch in total atom counts from different sources."


def assert_num_atom_per_token(features, token_budget):
    """Asserts that casting tokens to atoms results in the correct number of atoms."""

    per_atom_token_index = broadcast_token_feat_to_atoms(
        token_mask=features["token_mask"],
        num_atoms_per_token=features["num_atoms_per_token"],
        token_feat=features["token_index"],
    )
    start_atom_index_from_broadcast = torch.zeros(
        [token_budget], dtype=features["start_atom_index"].dtype
    )
    start_atom_index_from_broadcast_unpadded = torch.cat(
        [torch.zeros(1), torch.where(torch.diff(per_atom_token_index) != 0)[0] + 1]
    )
    start_atom_index_from_broadcast[
        : start_atom_index_from_broadcast_unpadded.shape[0]
    ] = start_atom_index_from_broadcast_unpadded

    assert torch.all(
        start_atom_index_from_broadcast == features["start_atom_index"]
    ), "Mismatch in number of atoms per token from broadcasting."


def assert_max_23_atoms_per_token(features):
    """Asserts that no token has more than 23 atoms."""
    assert ~(features["num_atoms_per_token"] > 23).any(), (
        "Token with more than 23 atoms found.",
    )


def assert_ligand_atomized(features):
    """Asserts that ligand tokens are atomized."""
    assert (features["is_ligand"] & features["is_atomized"])[
        features["is_ligand"].to(torch.bool)
    ].all(), ("Ligand token found that is not atomized.",)


def assert_atomized_one_atom(features):
    """Asserts that atomized tokens have only one atom."""
    assert (features["is_atomized"] & (features["num_atoms_per_token"] == 1))[
        features["is_atomized"].to(torch.bool)
    ].all(), "Atomized token with more than one atom found."


def assert_resid_asym_refuid_match(features):
    "Asserts that within the same residue_index-asym_id tuple, the ref_uid is the same."
    atom_resids = broadcast_token_feat_to_atoms(
        token_mask=features["token_mask"],
        num_atoms_per_token=features["num_atoms_per_token"],
        token_feat=features["residue_index"],
    )
    atom_asymids = broadcast_token_feat_to_atoms(
        token_mask=features["token_mask"],
        num_atoms_per_token=features["num_atoms_per_token"],
        token_feat=features["asym_id"],
    )

    unique_tuples = {}
    unique_id = 0
    result_ids = []

    # Iterate over elements of both tensors
    for a, b in zip(atom_resids, atom_asymids):
        tup = (a.item(), b.item())

        if tup not in unique_tuples:
            unique_tuples[tup] = unique_id
            unique_id += 1

        result_ids.append(unique_tuples[tup])

    # Convert list of ids to a tensor
    result_tensor = torch.tensor(result_ids)

    ref_space_uid_pos = torch.where(torch.diff(features["ref_space_uid"]) > 0)
    result_tensor_pos = torch.where(torch.diff(result_tensor) > 0)

    assert (
        torch.isin(ref_space_uid_pos[0], result_tensor_pos[0]).all()
    ), "Mismatch between changing positions of ref_space_uid and atom-broadcasted "
    "residue_index-asym_id tuples."


def assert_atom_pos_resolved(features):
    """Asserts that there are resolved atoms in the crop."""
    assert (
        ~(features["ground_truth"]["atom_resolved_mask"] == 0).all()
        | ~torch.isnan(features["ground_truth"]["atom_positions"]).all()
        | ~torch.isinf(features["ground_truth"]["atom_positions"]).all()
    ), "Atom positions are all-nan/inf or all atoms are unresolved."


def assert_one_entityid_per_asymid(features):
    """Asserts that there is only one entity_id per asym_id."""

    tups = set()
    for a, b in zip(features["asym_id"], features["entity_id"]):
        t = (a.item(), b.item())
        if ((t[0] != 0) & (t[1] != 0)) & (t not in tups):
            tups.add(t)

    assert len(tups) == len(
        set(features["asym_id"][features["asym_id"] != 0].tolist())
    ), "Multiple entity_ids per asym_id found."


def assert_no_identical_ref_pos(features):
    """Asserts that no two rows in ref_pos are identical."""

    ref_pos = features["ref_pos"]
    ref_pos_regular = ref_pos[
        ~(
            (ref_pos == 0).all(dim=1)
            | torch.isnan(ref_pos).all(dim=1)
            | torch.isinf(ref_pos).all(dim=1)
        )
    ]
    secondary_sort = ref_pos_regular[ref_pos_regular[:, 1].argsort(dim=0, stable=False)]
    primary_sort = secondary_sort[secondary_sort[:, 0].argsort(dim=0, stable=True)]
    assert torch.all(
        ~(torch.diff(primary_sort, dim=0) == 0).all(dim=1)
    ), "Found identical ref_pos coordinates."


def assert_no_all_zero_idxs(features):
    """Asserts none of the indexing tensors are all-zero."""
    index_keys = [
        "residue_index",
        "token_index",
        "asym_id",
        "entity_id",
        "sym_id",
        "restype",
        "is_protein",
        "is_rna",
        "is_dna",
        "is_ligand",
    ]
    for idx_key in index_keys:
        torch.any(~(features[idx_key] == 0)), f"Tensor '{idx_key}' contains all zeros."
    for idx_key in index_keys:
        (
            torch.any(~(features["ground_truth"][idx_key] == 0)),
            f"GT tensor '{idx_key}' contains all zeros.",
        )


def assert_gt_crop_slice(features):
    """Asserts that slicing the GT features creates matching target features."""
    cropped_token_index = features["token_index"]
    is_gt_in_crop = torch.isin(
        features["ground_truth"]["token_index"], cropped_token_index
    )

    for k in FEATURE_CORE_DTYPES:
        assert (
            features["ground_truth"][k][is_gt_in_crop] == features[k]
        ).all(), f"GT feature '{k}' does not match cropped feature."


def assert_shape(features, token_budget, n_templates):
    """Asserts the shape of the features."""
    for k, v in FULL_TOKEN_DIM_INDEX_MAP.items():
        for i in v:
            assert (
                features[k].shape[i] == token_budget
            ), f"Token shape mismatch for key '{k}'."
    for k, v in FULL_MSA_DIM_INDEX_MAP.items():
        for i in v:
            assert features[k].shape[i] <= 16384, f"MSA shape '{k}' larger than 16384."
    for k, v in FULL_TEMPLATE_DIM_INDEX_MAP.items():
        for i in v:
            assert (
                features[k].shape[i] == n_templates
            ), f"Template shape '{k}' shape mismatch."
    for k, v in FULL_OTHER_DIM_INDEX_MAP.items():
        for i in v:
            assert (
                features[k].shape[i] == FULL_OTHER_DIM_SIZE_MAP[k][i]
            ), f"Other shape '{k}' shape mismatch."


def assert_dtype(features):
    """Asserts the dtype of the features."""
    for k, v in (FEATURE_CORE_DTYPES | FEATURE_OTHER_DTYPES).items():
        assert features[k].dtype == v, f"Cropped feature '{k}' dtype mismatch."
    for k, v in (FEATURE_CORE_DTYPES | FEATURE_GT_DTYPES).items():
        assert (
            features["ground_truth"][k].dtype == v
        ), f"GT feature '{k}' dtype mismatch."
    for k, v in FEATURE_LOSS_DTYPES.items():
        assert (
            features["loss_weights"][k].dtype == v
        ), f"Loss feature '{k}' dtype mismatch."


def assert_resid_same_in_tokenid(features):
    """Asserts that the residue index doesn't change in each token."""
    residue_index_pos = torch.where(torch.diff(features["residue_index"]) > 0)
    token_index_pos = torch.where(torch.diff(features["token_index"]) > 0)

    assert torch.isin(
        residue_index_pos, token_index_pos
    ).all(), "Found residue indices that change within tokens."


def assert_all_unk_atomized(features):
    """Asserts that all tokens with unknown residue type are atomized."""
    is_unknown = features["restype"][:, 20] == 1
    assert (
        features["is_atomized"][is_unknown] == 1
    ).all(), "Found unknown residue tokens that are not atomized."
    assert (
        features["ground_truth"]["is_atomized"][is_unknown] == 1
    ).all(), "Found unknown GT residue tokens that are not atomized."


def assert_token_bonds_atomized(features):
    """Asserts that all tokens with token_bonds are atomized."""
    has_token_bonds_pos = torch.where(
        (
            torch.sum(features["token_bonds"], dim=0)
            + torch.sum(features["token_bonds"], dim=1)
        )
        > 0
    )[0]
    is_atomized_pos = torch.where(features["is_atomized"] == 1)[0]

    assert torch.isin(
        has_token_bonds_pos, is_atomized_pos
    ).all(), "Found unatomized tokens with token_bonds."


def assert_profile_sum(features):
    """Asserts that the sum of the profiles is 1 or 0."""
    profile_sum = torch.sum(features["profile"], dim=1)
    assert torch.all(
        torch.isclose(profile_sum, torch.tensor(1.0))
        | torch.isclose(profile_sum, torch.tensor(0.0))
    ), "Found profile column sums that are not 1 or 0."


FULL_TOKEN_DIM_INDEX_MAP = {
    "residue_index": [-1],
    "token_index": [-1],
    "asym_id": [-1],
    "entity_id": [-1],
    "sym_id": [-1],
    "restype": [-2],
    "is_protein": [-1],
    "is_rna": [-1],
    "is_dna": [-1],
    "is_ligand": [-1],
    "token_bonds": [-1, -2],
    "num_atoms_per_token": [-1],
    "is_atomized": [-1],
    "start_atom_index": [-1],
    "msa": [-2],
    "has_deletion": [-1],
    "deletion_value": [-1],
    "profile": [-2],
    "deletion_mean": [-1],
    "template_restype": [-2],
    "template_pseudo_beta_mask": [-1],
    "template_backbone_frame_mask": [-1],
    "template_distogram": [-2, -3],
    "template_unit_vector": [-2, -3],
}
FULL_MSA_DIM_INDEX_MAP = {
    "msa": [-3],
    "has_deletion": [-2],
    "deletion_value": [-2],
}
FULL_TEMPLATE_DIM_INDEX_MAP = {
    "template_restype": [-3],
    "template_pseudo_beta_mask": [-2],
    "template_backbone_frame_mask": [-2],
    "template_distogram": [-4],
    "template_unit_vector": [-4],
}
FULL_OTHER_DIM_INDEX_MAP = {
    "restype": [-1],
    "ref_pos": [-1],
    "ref_element": [-1],
    "ref_atom_name_chars": [-1, -2],
    "msa": [-1],
    "template_restype": [-1],
    "template_distogram": [-1],
    "template_unit_vector": [-1],
}
FULL_OTHER_DIM_SIZE_MAP = {
    "restype": [32],
    "ref_pos": [3],
    "ref_element": [119],
    "ref_atom_name_chars": [4, 64],
    "msa": [32],
    "template_restype": [32],
    "template_distogram": [39],
    "template_unit_vector": [3],
}

FEATURE_CORE_DTYPES = {
    "residue_index": torch.int32,
    "token_index": torch.int32,
    "asym_id": torch.int32,
    "entity_id": torch.int32,
    "sym_id": torch.int32,
    "restype": torch.int32,
    "is_protein": torch.int32,
    "is_rna": torch.int32,
    "is_dna": torch.int32,
    "is_ligand": torch.int32,
    "token_bonds": torch.int32,
    "num_atoms_per_token": torch.int32,
    "is_atomized": torch.int32,
    "start_atom_index": torch.int32,
    "token_mask": torch.float32,
}
FEATURE_GT_DTYPES = {
    "atom_positions": torch.float32,
    "atom_resolved_mask": torch.float32,
}
FEATURE_OTHER_DTYPES = {
    "ref_pos": torch.float32,
    "ref_mask": torch.int32,
    "ref_element": torch.int32,
    "ref_charge": torch.float32,
    "ref_atom_name_chars": torch.int32,
    "ref_space_uid": torch.int32,
    "msa": torch.int32,
    "has_deletion": torch.float32,
    "deletion_value": torch.float32,
    "profile": torch.float32,
    "deletion_mean": torch.float32,
    "template_restype": torch.int32,
    "template_pseudo_beta_mask": torch.float32,
    "template_backbone_frame_mask": torch.float32,
    "template_distogram": torch.int32,
    "template_unit_vector": torch.float32,
    "msa_mask": torch.float32,
    "num_recycles": torch.int32,
    "num_paired_seqs": torch.int32,
}
FEATURE_LOSS_DTYPES = {
    "bond": torch.float32,
    "smooth_lddt": torch.float32,
    "mse": torch.float32,
    "distogram": torch.float32,
    "experimentally_resolved": torch.float32,
    "plddt": torch.float32,
    "pae": torch.float32,
    "pde": torch.float32,
}


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.run_asserts = run_asserts
        self.save_features = save_features
        self.save_atom_array = save_atom_array
        self.save_full_traceback = save_full_traceback
        self.save_statistics = save_statistics

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
                    structure_format="bcif",
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

            # Save extra data
            if self.save_statistics:
                self.save_extra_data(
                    pdb_id,
                    preferred_chain_or_interface,
                    features,
                    atom_array_cropped,
                    atom_array,
                )

            # Asserts
            if self.run_asserts:
                assert_no_nan_inf(features)
                assert_token_atom_sum_match(features)
                assert_num_atom_per_token(features, self.token_budget)
                assert_max_23_atoms_per_token(features)
                assert_ligand_atomized(features)
                assert_atomized_one_atom(features)
                assert_resid_asym_refuid_match(features)
                assert_atom_pos_resolved(features)
                assert_one_entityid_per_asymid(features)
                assert_no_identical_ref_pos(features)
                assert_no_all_zero_idxs(features)
                assert_gt_crop_slice(features)
                assert_shape(features, self.token_budget, self.n_templates)
                assert_dtype(features)
                assert_all_unk_atomized(features)
                assert_token_bonds_atomized(features)
                assert_profile_sum(features)

            return features
        except AssertionError as e:
            # Catch assertion errors
            self.logger.error(
                f"ASSERTION ERROR processing datapoint {index}, PDB ID: {pdb_id}"
            )
            self.logger.error(f"Error message: {e}")

            # Save features, atom array and per sample traceback
            self.save_features_atom_array(
                features, atom_array_cropped, pdb_id, preferred_chain_or_interface
            )
            self.save_full_traceback_for_sample(e, pdb_id, preferred_chain_or_interface)
            return features

        except Exception as e:
            # Catch all other errors
            self.logger.error(
                f"OTHER ERROR processing datapoint {index}, PDB ID: {pdb_id}"
            )
            self.logger.error(f"Error message: {e}")

            # Save features and atom array
            self.save_features_atom_array(
                features, atom_array_cropped, pdb_id, preferred_chain_or_interface
            )
            self.save_full_traceback_for_sample(e, pdb_id, preferred_chain_or_interface)

            return features

    def save_features_atom_array(
        self, features, atom_array_cropped, pdb_id, preferred_chain_or_interface
    ):
        """Saves features and atom array from the worker process to disk."""
        if self.save_features | self.save_atom_array:
            log_output_dir = self.logger.extra["log_output_directory"] / Path(
                "worker_{}/{}".format(self.logger.extra["worker_id"], pdb_id)
            )
            log_output_dir.mkdir(parents=True, exist_ok=True)

        preferred_chain_or_interface = (
            "-".join(preferred_chain_or_interface)
            if isinstance(preferred_chain_or_interface, list)
            else preferred_chain_or_interface
        )
        if self.save_features:
            torch.save(
                features,
                log_output_dir
                / Path(f"{pdb_id}-{preferred_chain_or_interface}_features.pt"),
            )
        if self.save_atom_array & (atom_array_cropped is not None):
            with open(
                log_output_dir
                / Path(f"{pdb_id}-{preferred_chain_or_interface}_atom_array.pkl"),
                "wb",
            ) as f:
                pkl.dump(atom_array_cropped, f)
            # strucio.save_structure(
            #     log_output_dir
            #     / Path(f"{pdb_id}-{preferred_chain_or_interface}_atom_array.cif"),
            #     atom_array_cropped,
            # )

    def save_full_traceback_for_sample(self, e, pdb_id, preferred_chain_or_interface):
        """Saves the full traceback to for failed samples."""
        if self.save_full_traceback:
            log_output_dir = self.logger.extra["log_output_directory"] / Path(
                "worker_{}/{}".format(self.logger.extra["worker_id"], pdb_id)
            )
            preferred_chain_or_interface = (
                "-".join(preferred_chain_or_interface)
                if isinstance(preferred_chain_or_interface, list)
                else preferred_chain_or_interface
            )
            sample_logger = logging.getLogger(
                f"{pdb_id}-{preferred_chain_or_interface}"
            )
            if sample_logger.hasHandlers():
                sample_logger.handlers.clear()
            sample_logger.setLevel(self.logger.logger.level)
            sample_logger.propagate = False
            sample_file_handler = logging.FileHandler(
                log_output_dir
                / Path(f"{pdb_id}-{preferred_chain_or_interface}_error.log"),
                mode="w",
            )
            sample_file_handler.setLevel(self.logger.logger.level)
            sample_logger.addHandler(sample_file_handler)

            sample_logger.info(
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

    def save_extra_data(
        self,
        pdb_id,
        preferred_chain_or_interface,
        features,
        atom_array_cropped,
        atom_array,
    ):
        """Saves additional data statistics."""
        if (self.save_statistics is not None) & (len(self.save_statistics) > 0):
            log_output_dir = self.logger.extra["log_output_directory"] / Path(
                "worker_{}".format(self.logger.extra["worker_id"])
            )
            preferred_chain_or_interface = (
                "-".join(preferred_chain_or_interface)
                if isinstance(preferred_chain_or_interface, list)
                else preferred_chain_or_interface
            )
            line = f"{pdb_id}\t{preferred_chain_or_interface}\t"

            atom_array_protein = atom_array[
                atom_array.molecule_type_id != MoleculeType.PROTEIN
            ]
            atom_array_protein_cropped = atom_array_cropped[
                atom_array_cropped.molecule_type_id != MoleculeType.PROTEIN
            ]
            atom_array_rna = atom_array[atom_array.molecule_type_id != MoleculeType.RNA]
            atom_array_rna_cropped = atom_array_cropped[
                atom_array_cropped.molecule_type_id != MoleculeType.RNA
            ]
            atom_array_dna = atom_array[atom_array.molecule_type_id != MoleculeType.DNA]
            atom_array_dna_cropped = atom_array_cropped[
                atom_array_cropped.molecule_type_id != MoleculeType.DNA
            ]
            atom_array_ligand = atom_array[
                atom_array.molecule_type_id != MoleculeType.LIGAND
            ]
            atom_array_ligand_cropped = atom_array_cropped[
                atom_array_cropped.molecule_type_id != MoleculeType.LIGAND
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

            statistics = []

            # Number of atoms
            for aa in all_aa:
                statistics += [len(aa)]

            # Number of residues
            for aa in polymer_aa:
                resid_tensor = torch.tensor(aa.res_id)
                statistics += [len(torch.unique_consecutive(resid_tensor))]

            for aa in full_aa:
                # Number of unresolved atoms
                statistics += [np.isnan(aa.coord).any(axis=1).sum()]
                # Number of unresolved residues
                cumsums = np.cumsum(np.isnan(aa.coord).any(axis=1))
                statistics += [(np.diff(cumsums[residue_starts]) > 0).sum()]

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
            statistics += [(msa.sum(dim=0)[:, -1] < msa.size(0)).sum()]

            # number of templates
            tbfm = features["template_backbone_frame_mask"]
            statistics += [(tbfm == 1).any(dim=-1).sum().item()]
            # number of tokens with any aligned template columns in the crop
            statistics += [(tbfm == 1).any(dim=-2).sum().item()]

            # number of tokens
            for aa in full_aa:
                statistics += [len(set(aa.token_id))]

            for aa in polymer_aa:
                # number of residue tokens atomized due to special
                if aa[0].molecule_type_id == MoleculeType.PROTEIN:
                    vocab = STANDARD_PROTEIN_RESIDUES_3
                elif aa[0].molecule_type_id == MoleculeType.RNA:
                    vocab = STANDARD_RNA_RESIDUES
                else:
                    vocab = STANDARD_DNA_RESIDUES
                is_special_aa = ~np.isin(aa.res_name, vocab)
                rs = struc.get_residue_starts(aa)
                statistics += [is_special_aa[rs].sum()]

                # number of residue tokens atomized due to covalent modifications
                is_standard_atomized_aa = (~is_special_aa) & aa.is_atomized
                statistics += [is_standard_atomized_aa[rs].sum()]

            # radius of gyration
            for aa in full_aa:
                aa_resolved = aa[~np.isnan(aa.coord).any(axis=1)]
                statistics += [struc.gyration_radius(aa_resolved)]

            # interface statistics
            for aa_a, aa_b in zip(
                [
                    atom_array_protein,
                    atom_array_protein_cropped,
                ]
                * 4,
                per_moltype_aa,
            ):
                statistics += [self.get_interface_string(aa_a, aa_b)]

            # sub-pipeline runtimes - add via extra arguments

            # Collate into tab format
            line += "\t".join(map(str, statistics))
            line += "\n"

            with open(
                log_output_dir / Path("extra_data.txt"),
                "a",
            ) as f:
                f.write(line)

    @staticmethod
    def compute_interface(
        query_atom_array: AtomArray,
        target_atom_array: AtomArray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the atom/chain pairs that form the interface between two structures.

        Optionally returns the residue pairs.

        Args:
            query_atom_array (AtomArray):
                Query atom array.
            target_atom_array (AtomArray):
                Target atom array.
            return_res_pairs (bool, optional):
                Whether to return an array of residue pairs. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                tuple of residue pairs and chain pairs
        """
        atom_pairs, chain_pairs = get_query_interface_atom_pair_idxs(
            query_atom_array,
            target_atom_array,
            distance_threshold=5.0,
            return_chain_pairs=True,
        )
        res_pairs = np.concatenate(
            [
                query_atom_array.res_id[atom_pairs[:, 0]][..., np.newaxis],
                target_atom_array.res_id[atom_pairs[:, 1]][..., np.newaxis],
            ],
            axis=1,
        )
        return res_pairs, chain_pairs

    @staticmethod
    def encode_interface(res_pairs: np.ndarray, chain_pairs: np.ndarray) -> str:
        """Encodes the interface as a string.

        Args:
            res_pairs (np.ndarray):
                Array of residue index pairs.
            chain_pairs (np.ndarray):
                Array of chain index pairs.

        Returns:
            str:
                Encoded interface string. Has the format:
                "chain_id.res_id-chain_id.res_id;..."
        """
        unique_contacts = np.unique(
            np.concatenate([res_pairs, chain_pairs], axis=-1), axis=0
        )
        return ";".join(
            np.core.defchararray.add(
                np.core.defchararray.add(
                    np.core.defchararray.add(
                        np.core.defchararray.add(unique_contacts[:, 2], "."),
                        unique_contacts[:, 0],
                    ),
                    "-",
                ),
                np.core.defchararray.add(
                    np.core.defchararray.add(unique_contacts[:, 3], "."),
                    unique_contacts[:, 1],
                ),
            )
        )

    @staticmethod
    def get_interface_string(
        self, query_atom_array: AtomArray, target_atom_array: AtomArray
    ) -> str:
        """Computes the interface string between two structures.

        Args:
            query_atom_array (AtomArray):
                Query atom array.
            target_atom_array (AtomArray):
                Target atom array.
        Returns:
            str:
                Encoded interface string
        """
        r, c = self.compute_interface(
            query_atom_array, target_atom_array, return_res_pairs=True
        )
        return self.encode_interface(r, c)

    @staticmethod
    def decode_interface(interface_string: str) -> tuple[np.ndarray, np.ndarray]:
        """Decodes the interface string.

        Args:
            interface (str):
                Encoded interface string. Has the format:
                "chain_id.res_id-chain_id.res_id;..."

        Returns:
            tuple[np.ndarray, np.ndarray]:
                Array of residue index pairs and chain index pairs.
        """
        contacts = interface_string.split(";")
        contacts = np.array([c.split("-") for c in contacts])
        contacts = np.char.split(contacts, ".")
        contacts_flattened = np.concatenate(contacts.ravel())

        chains = contacts_flattened[::2]
        residues = np.array(contacts_flattened[1::2], dtype=int).reshape(-1, 2)

        chain_residues = np.column_stack((residues, chains.reshape(-1, 2)))

        return chain_residues[:, :2], chain_residues[:, 2:]


@click.command()
@click.option(
    "--runner-yml-file",
    required=True,
    help="Yaml that specifies model and dataset parameters," "see examples/runner.yml",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--seed",
    required=True,
    help="Seed for reproducibility",
    type=int,
)
@click.option(
    "--with-model-fwd",
    required=True,
    help="Whether to run the model forward pass with the produced features",
    type=bool,
)
@click.option(
    "--log-output-directory",
    required=True,
    help="Path to directory where logs will be saved",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=True),
    help="Set the logging level",
)
@click.option(
    "--run-asserts",
    default=True,
    type=bool,
    help="Whether to run asserts.",
)
@click.option(
    "--save-features",
    default=True,
    type=bool,
    help="Whether to run asserts.",
)
@click.option(
    "--save-atom-array",
    default=True,
    type=bool,
    help="Whether to run asserts.",
)
@click.option(
    "--save-full-traceback",
    default=True,
    type=bool,
    help="Whether to save the per-sample full tracebacks.",
)
@click.option(
    "--save-statistics",
    type=bool,
    default=False,
    help="Additional data to save during data processing.",
)
def main(
    runner_yml_file: Path,
    seed: int,
    with_model_fwd: bool,
    log_output_directory: Path,
    log_level: str,
    run_asserts: bool,
    save_features: bool,
    save_atom_array: bool,
    save_full_traceback: bool,
    save_statistics: list[str],
) -> None:
    """Main function for running the data pipeline treadmill.

    Args:
        runner_yml_file (Path):
            File path to the input yaml file.
        seed (int):
            Seed to use for data pipeline.
        with_model_fwd (bool):
            Whether to run the model forward pass with the produced features.
        log_level (str):
            Logging level.
        run_asserts (bool):
            Whether to run asserts.
        save_features (bool):
            Whether to save features when an exception occurs.
        save_atom_array (bool):
            Whether to save atom array when an exception occurs.
        save_full_traceback (bool):
            Whether to save the per-sample full traceback when an exception occurs.
        save_statistics (list[str]):
            Additional data to save during data processing.

    Raises:
        ValueError:
            If num_workers < 1.
        NotImplementedError:
            If with_model_fwd is True.

    Returns:
        None
    """
    # Set seed
    pl.seed_everything(seed, workers=False)

    # Parse runner yml file and init Dataset
    runner_args = ConfigDict(config_utils.load_yaml(runner_yml_file))

    if runner_args.num_workers < 1:
        raise ValueError("This script only works with num_workers >= 1.")

    project_entry = registry.get_project_entry(runner_args.project_type)
    project_config = registry.make_config_with_presets(
        project_entry, runner_args.presets
    )
    dataset_config_builder = project_entry.dataset_config_builder
    data_module_config = registry.make_dataset_module_config(
        runner_args,
        dataset_config_builder,
        project_config,
    )
    dataset = WeightedPDBDatasetWithLogging(
        run_asserts=run_asserts,
        save_features=save_features,
        save_atom_array=save_atom_array,
        save_full_traceback=save_full_traceback,
        save_statistics=save_statistics,
        dataset_config=data_module_config.datasets[0].config,
    )

    # Configure worker init function logger
    def configure_worker_init_func_logger(worker_id: int) -> logging.Logger:
        # Configure logging
        worker_logger = logging.getLogger()
        numeric_level = getattr(logging, log_level)
        worker_logger.setLevel(numeric_level)

        # Clear any existing handlers
        if worker_logger.hasHandlers():
            worker_logger.handlers.clear()

        # Create a handler for each worker and corresponding dir
        worker_dir = log_output_directory / Path(f"worker_{worker_id}")
        worker_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(worker_dir / Path(f"worker_{worker_id}.log"))
        formatter = logging.Formatter(
            "%(asctime)s - Worker %(worker_id)s - " "%(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        worker_logger.addHandler(handler)

        # Add worker_id and log_output_directory to the logger (for formatting)
        worker_logger = logging.LoggerAdapter(
            worker_logger,
            {"worker_id": worker_id, "log_output_directory": log_output_directory},
        )

        # Set the logger to the local copy of the dataset in the current worker
        worker_info = get_worker_info()
        worker_dataset = worker_info.dataset
        worker_dataset.logger = worker_logger
        return worker_logger

    # Set up extra data file
    def configure_extra_data_file(worker_id: int) -> None:
        """Configures the file to save extra data."""
        all_headers = [
            "pdb-id",
            "chain-or-interface",
            "atoms",
            "atoms-crop",
            "atoms-protein",
            "atoms-protein-crop",
            "atoms-rna",
            "atoms-rna-crop",
            "atoms-dna",
            "atoms-dna-crop",
            "atoms-ligand",
            "atoms-ligand-crop",
            "res-protein",
            "res-protein-crop",
            "res-rna",
            "res-rna-crop",
            "res-dna",
            "res-dna-crop",
            "atoms-unresolved",
            "atoms-unresolved-crop",
            "res-unresolved",
            "res-unresolved-crop",
            "chains",
            "chains-crop",
            "entities-protein",
            "entities-protein-crop",
            "entities-rna",
            "entities-rna-crop",
            "entities-dna",
            "entities-dna-crop",
            "entities-ligand",
            "entities-ligand-crop",
            "msa-depth",
            "msa-num-paired-seqs",
            "msa-aligned-cols",
            "templates",
            "templates-aligned-cols",
            "tokens",
            "tokens-crop",
            "res-special-protein",
            "res-special-protein-crop",
            "res-special-rna",
            "res-special-rna-crop",
            "res-special-dna",
            "res-special-dna-crop",
            "res-covmod-protein",
            "res-covmod-protein-crop",
            "res-covmod-rna",
            "res-covmod-rna-crop",
            "res-covmod-dna",
            "res-covmod-dna-crop",
            "gyration-radius",
            "gyration-radius-crop",
            "interface-protein-protein",
            "interface-protein-rna",
            "interface-protein-dna",
            "interface-protein-ligand",
        ]
        extra_data_file = log_output_directory / Path(
            f"worker_{worker_id}/extra_data.txt"
        )
        with open(extra_data_file, "w") as f:
            f.write("\t".join(all_headers) + "\n")

    # Set up custom worker init function with logging
    def worker_init_function_with_logging(
        worker_id: int, rank: int | None = None
    ) -> None:
        """Modified default Lightning worker_init_fn with manual data seed.

        This worker_init_fn enables decoupling stochastic processes in the data
        pipeline from those in the model. Taken from Pytorch Lightning 2.4.1 source
        code: https://github.com/Lightning-AI/pytorch-lightning/blob/f3f10d460338ca8b2901d5cd43456992131767ec/src/lightning/fabric/utilities/seed.py#L85

        Args:
            worker_id (int):
                Worker id.
            rank (Optional[int], optional):
                Worker process rank. Defaults to None.
        """
        # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
        global_rank = rank if rank is not None else rank_zero_only.rank
        process_seed = torch.initial_seed()
        # back out the base seed so we can use all the bits
        base_seed = process_seed - worker_id
        seed_sequence = _generate_seed_sequence(
            base_seed, worker_id, global_rank, count=4
        )
        torch.manual_seed(seed_sequence[0])  # torch takes a 64-bit seed
        random.seed(
            (seed_sequence[1] << 32) | seed_sequence[2]
        )  # combine two 64-bit seeds
        if _NUMPY_AVAILABLE:
            import numpy as np

            np.random.seed(
                seed_sequence[3] & 0xFFFFFFFF
            )  # numpy takes 32-bit seed only

        # Configure logger and log process & worker IDs
        worker_logger = configure_worker_init_func_logger(worker_id)
        worker_logger.info("Worker init function completed.")
        worker_logger.info(
            "logger worker ID: {}".format(worker_logger.extra["worker_id"])
        )
        worker_logger.info(f"process ID: {os.getpid()}")

        # Configure extra data file
        configure_extra_data_file(worker_id)

    # Configure DataLoader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=data_module_config.batch_size,
        num_workers=data_module_config.num_workers,
        worker_init_fn=worker_init_function_with_logging,
    )

    # Init model
    if with_model_fwd:
        raise NotImplementedError(
            "Running the treadmill script with model forward pass"
            " is not yet implemented."
        )

    # Iterate over dataset
    try:
        for _ in tqdm(
            data_loader, desc="Iterating over WeightedPDBDataset", total=len(dataset)
        ):
            pass
    finally:
        # Collate the extra data from different workers
        if save_statistics:
            df_all = pd.DataFrame()
            for worker_id in range(runner_args.num_workers):
                worker_extra_data_file = log_output_directory / Path(
                    f"worker_{worker_id}/extra_data.txt"
                )
                df_all = pd.concat(
                    [df_all, pd.read_csv(worker_extra_data_file, sep="\t")]
                )
                worker_extra_data_file.unlink()

            df_all.to_csv(log_output_directory / Path("extra_data.txt"), sep="\t")


if __name__ == "__main__":
    main()

# TODOs:
# 5. add pytorch profiler and test, test with py-spy - should be separate runs from the
# debugging runs
# (6. implement the model forward pass)
# 7. add logic to save which PDB entries/chains were already tested and restart from
# there
# 8. Add logic to re-crop the structure if the number of tokens is larger than the
# token budget - the number of re-crops and featurizations should be determined
# dynamically and in a way that likely covers the entire structure but with a
# maximun number of re-crops
# 9. Add logic to save extra data
