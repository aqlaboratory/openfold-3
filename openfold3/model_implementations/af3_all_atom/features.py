import ml_collections as mlc

NUM_TOKEN = "num token placeholder"
NUM_ATOM = "num atom placeholder"
NUM_MSA = "num seq in msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

feature_dict = mlc.ConfigDict(
    {
        "feat": {
            # Indexing
            "residue_index": [NUM_TOKEN],
            "token_index": [NUM_TOKEN],
            "asym_id": [NUM_TOKEN],
            "entity_id": [NUM_TOKEN],
            "sym_id": [NUM_TOKEN],
            "restype": [NUM_TOKEN, None],
            "is_protein": [NUM_TOKEN],
            "is_rna": [NUM_TOKEN],
            "is_dna": [NUM_TOKEN],
            "is_ligand": [NUM_TOKEN],
            # Reference conformers
            "ref_pos": [NUM_ATOM, None],
            "ref_mask": [NUM_ATOM],
            "ref_element": [NUM_ATOM, None],
            "ref_charge": [NUM_ATOM],
            "ref_atom_name_chars": [NUM_ATOM, None, None],
            "ref_space_uid": [NUM_ATOM],
            # MSA
            "msa": [NUM_MSA, NUM_TOKEN, None],
            "has_deletion": [NUM_MSA, NUM_TOKEN],
            "deletion_value": [NUM_MSA, NUM_TOKEN],
            "profile": [NUM_TOKEN, None],
            "deletion_mean": [NUM_TOKEN],
            # Templates
            "template_restype": [NUM_TEMPLATES, NUM_TOKEN, None],
            "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_TOKEN],
            "template_backbone_frame_mask": [NUM_TEMPLATES, NUM_TOKEN],
            "template_distogram": [NUM_TEMPLATES, NUM_TOKEN, NUM_TOKEN, None],
            "template_unit_vector": [NUM_TEMPLATES, NUM_TOKEN, NUM_TOKEN, None],
            # Bonding
            "token_bonds": [NUM_TOKEN, NUM_TOKEN],
            # Ground truth
            "gt_pos": [NUM_TOKEN, None],
            "exp_resolved_mask": [NUM_TOKEN],
            # Atomization
            "num_atoms_per_token": [NUM_TOKEN],
            "start_atom_index": [NUM_TOKEN],
            "is_atomized": [NUM_TOKEN],
            # Loss switches
            "apply_plddt_loss": [None],
            "apply_pde_loss": [None],
            "apply_exp_resolved_loss": [None],
            "apply_pae_loss": [None],
            "apply_mse_loss": [None],
            "apply_bond_loss": [None],
            "apply_slddt_loss": [None],
            "apply_distogram_loss": [None],
        },
    }
)
