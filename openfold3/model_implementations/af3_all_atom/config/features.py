import ml_collections as mlc

NUM_TOKENS = "num tokens placeholder"
NUM_ATOMS = "num atoms placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

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
            "msa_mask": [NUM_MSA_SEQ, NUM_TOKENS],
            "num_main_msa_seqs": [],
            "gt_atom_positions": [NUM_ATOMS, 3],
            "gt_atom_mask": [NUM_ATOMS],
            "resolution": [],
            "is_distillation": [],
        },
    }
)
