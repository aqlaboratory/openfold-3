import ml_collections as mlc

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

feature_dict = mlc.ConfigDict(
    {
        "feat": {
            "aatype": [NUM_RES],
            "all_atom_mask": [NUM_RES, None],
            "all_atom_positions": [NUM_RES, None, None],
            # TODO: Resolve missing features, remove processed msa feats
            # "all_chains_entity_ids": [],
            # "all_crops_all_chains_mask": [],
            # "all_crops_all_chains_positions": [],
            # "all_crops_all_chains_residue_ids": [],
            "assembly_num_chains": [],
            "asym_id": [NUM_RES],
            "atom14_atom_exists": [NUM_RES, None],
            "atom37_atom_exists": [NUM_RES, None],
            "bert_mask": [NUM_MSA_SEQ, NUM_RES],
            "cluster_bias_mask": [NUM_MSA_SEQ],
            "cluster_profile": [NUM_MSA_SEQ, NUM_RES, None],
            "cluster_deletion_mean": [NUM_MSA_SEQ, NUM_RES],
            "deletion_matrix": [NUM_MSA_SEQ, NUM_RES],
            "deletion_mean": [NUM_RES],
            "entity_id": [NUM_RES],
            "entity_mask": [NUM_RES],
            "extra_deletion_matrix": [NUM_EXTRA_SEQ, NUM_RES],
            "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
            "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
            # "mem_peak": [],
            "msa": [NUM_MSA_SEQ, NUM_RES],
            "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
            "msa_mask": [NUM_MSA_SEQ, NUM_RES],
            "msa_profile": [NUM_RES, None],
            "num_alignments": [],
            "num_templates": [],
            # "queue_size": [],
            "residue_index": [NUM_RES],
            "residx_atom14_to_atom37": [NUM_RES, None],
            "residx_atom37_to_atom14": [NUM_RES, None],
            "resolution": [],
            "seq_length": [],
            "seq_mask": [NUM_RES],
            "sym_id": [NUM_RES],
            "target_feat": [NUM_RES, None],
            "template_aatype": [NUM_TEMPLATES, NUM_RES],
            "template_all_atom_mask": [NUM_TEMPLATES, NUM_RES, None],
            "template_all_atom_positions": [
                NUM_TEMPLATES,
                NUM_RES,
                None,
                None,
            ],
            "true_msa": [NUM_MSA_SEQ, NUM_RES],
        },
    }
)
