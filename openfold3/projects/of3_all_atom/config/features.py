# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections as mlc

NUM_TOKENS = "num tokens placeholder"
NUM_ATOMS = "num atoms placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_TEMPLATES = "num templates placeholder"


# TODO: Add new permutation features
feature_dict = mlc.ConfigDict(
    {
        "feat": {
            "residue_index": [NUM_TOKENS],
            "token_index": [NUM_TOKENS],
            "asym_id": [NUM_TOKENS],
            "entity_id": [NUM_TOKENS],
            "sym_id": [NUM_TOKENS],
            "restype": [NUM_TOKENS, 32],
            "is_protein": [NUM_TOKENS],
            "is_rna": [NUM_TOKENS],
            "is_dna": [NUM_TOKENS],
            "is_ligand": [NUM_TOKENS],
            "is_atomized": [NUM_TOKENS],
            "ref_pos": [NUM_ATOMS, 3],
            "ref_mask": [NUM_ATOMS],
            "ref_element": [NUM_ATOMS, 128],
            "ref_charge": [NUM_ATOMS],
            "ref_atom_name_chars": [NUM_ATOMS, 4, 64],
            "ref_space_uid": [NUM_ATOMS],
            "msa": [NUM_MSA_SEQ, NUM_TOKENS, 32],
            "has_deletion": [NUM_MSA_SEQ, NUM_TOKENS],
            "deletion_value": [NUM_MSA_SEQ, NUM_TOKENS],
            "profile": [NUM_TOKENS, 32],
            "deletion_mean": [NUM_TOKENS],
            "template_restype": [NUM_TEMPLATES, NUM_TOKENS, 32],
            "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_TOKENS],
            "template_backbone_frame_mask": [NUM_TEMPLATES, NUM_TOKENS],
            "template_distogram": [NUM_TEMPLATES, NUM_TOKENS, NUM_TOKENS, 39],
            "template_unit_vector": [NUM_TEMPLATES, NUM_TOKENS, NUM_TOKENS, 3],
            "token_bonds": [NUM_TOKENS, NUM_TOKENS],
            # Features not included in AF3 docs
            "num_atoms_per_token": [NUM_TOKENS],
            "start_atom_index": [NUM_TOKENS],
            "token_mask": [NUM_TOKENS],
            "atom_mask": [NUM_ATOMS],
            "atom_to_token_index": [NUM_ATOMS],
            "msa_mask": [NUM_MSA_SEQ, NUM_TOKENS],
            "num_paired_seqs": [],
            # Permutation alignment features
            "mol_entity_id": [NUM_TOKENS],
            "mol_sym_id": [NUM_TOKENS],
            "mol_sym_token_index": [NUM_TOKENS],
            "mol_sym_component_id": [NUM_TOKENS],
            "ground_truth": {
                "token_index": [NUM_TOKENS],
                "restype": [NUM_TOKENS, 32],
                "is_protein": [NUM_TOKENS],
                "is_rna": [NUM_TOKENS],
                "is_dna": [NUM_TOKENS],
                "is_ligand": [NUM_TOKENS],
                "is_atomized": [NUM_TOKENS],
                "token_mask": [NUM_TOKENS],
                "num_atoms_per_token": [NUM_TOKENS],
                "start_atom_index": [NUM_TOKENS],
                "atom_mask": [NUM_ATOMS],
                "atom_positions": [NUM_ATOMS, 3],
                "atom_resolved_mask": [NUM_ATOMS],
                # Permutation alignment features
                "mol_entity_id": [NUM_TOKENS],
                "mol_sym_id": [NUM_TOKENS],
                "mol_sym_token_index": [NUM_TOKENS],
                "mol_sym_component_id": [NUM_TOKENS],
                "intra_filter_atomized": [NUM_ATOMS],  # Validation only
                "inter_filter_atomized": [NUM_ATOMS, NUM_ATOMS],  # Validation only
            },
            "loss_weights": {
                "bond": [],
                "smooth_lddt": [],
                "mse": [],
                "plddt": [],
                "pde": [],
                "experimentally_resolved": [],
                "pae": [],
                "distogram": [],
                "disable_non_protein_diffusion_weights": [],
            },
        },
    }
)
