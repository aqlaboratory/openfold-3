import ml_collections as mlc

import openfold3.model_implementations.af2_monomer.config.linear_init_config as lin_init
import openfold3.model_implementations.af2_monomer.config.base_config as af2_monomer_config
import openfold3.model_implementations.af2_multimer.config.linear_init_config as lin_init_mult
from openfold3.core.config.config_utils import update_config_dict
from openfold3.model_implementations.af2_multimer.config.features import (
    feature_dict as multimer_feature_dict,
)

c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)

blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
templates_enabled = mlc.FieldReference(True, field_type=bool)
embed_template_torsion_angles = mlc.FieldReference(True, field_type=bool)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

# TODO(Jennifer): Make single multimer config, rather than updating from monomer
monomer_config = af2_monomer_config.config

multimer_config_update = mlc.ConfigDict(
    {
        "globals": {"is_multimer": True},
        "data": {
            "common": {
                "feat": multimer_feature_dict,
                "max_recycling_iters": 20,  # For training, value is 3
                "unsupervised_features": [
                    "aatype",
                    "residue_index",
                    "msa",
                    "num_alignments",
                    "seq_length",
                    "between_segment_residues",
                    "deletion_matrix",
                    "no_recycling_iters",
                    # Additional multimer features
                    "msa_mask",
                    "seq_mask",
                    "asym_id",
                    "entity_id",
                    "sym_id",
                ],
            },
            "supervised": {"clamp_prob": 1.0},
            # TODO: Change max_msa_clusters and max_extra_msa to multimer feats within
            # model:
            # c.model.input_embedder.num_msa = 508
            # c.model.extra_msa.extra_msa_embedder.num_extra_msa = 2048
            "predict": {"max_msa_clusters": 508, "max_extra_msa": 2048},
            "eval": {"max_msa_clusters": 508, "max_extra_msa": 2048},
            "train": {
                "max_msa_clusters": 508,
                "max_extra_msa": 2048,
                "block_delete_msa": False,
                "crop_size": 640,
                "spatial_crop_prob": 0.5,
                "interface_threshold": 10.0,
                "clamp_prob": 1.0,
            },
        },
        "model": {
            "input_embedder": {
                "tf_dim": 21,
                # "num_msa": 508,
                "max_relative_chain": 2,
                "max_relative_idx": 32,
                "use_chain_relative": True,
                "linear_init_params": lin_init_mult.input_emb_init,
            },
            "template": {
                "template_single_embedder": {
                    "c_in": 34,
                    "c_out": c_m,
                    "linear_init_params": lin_init_mult.templ_single_feat_emb_init,
                },
                "template_pair_embedder": {
                    "c_in": c_z,
                    "c_out": c_t,
                    "c_dgram": 39,
                    "c_aatype": 22,
                    "linear_init_params": lin_init_mult.templ_pair_feat_emb_init,
                },
                "template_pair_stack": {
                    "tri_mul_first": True,
                    "fuse_projection_weights": True,
                    "linear_init_params": lin_init_mult.pair_block_init,
                },
                "c_t": c_t,
                "c_z": c_z,
                "use_unit_vector": True,
                "linear_init_params": lin_init_mult.templ_module_init,
            },
            "extra_msa": {
                # "extra_msa_embedder": {
                #     "num_extra_msa": 2048
                # },
                "extra_msa_stack": {
                    "opm_first": True,
                    "fuse_projection_weights": True,
                    "linear_init_params": lin_init_mult.extra_msa_block_init,
                },
            },
            "evoformer_stack": {
                "opm_first": True,
                "fuse_projection_weights": True,
                "linear_init_params": lin_init_mult.evo_block_init,
            },
            "structure_module": {
                "trans_scale_factor": 20,
                "linear_init_params": lin_init_mult.structure_module_init,
            },
            "heads": {
                "tm": {"ptm_weight": 0.2, "iptm_weight": 0.8, "enabled": True},
                "masked_msa": {"c_out": 22},
            },
            "recycle_early_stop_tolerance": 0.5,  # For training, value is -1.
        },
        "loss": {
            "fape": {
                "intra_chain_backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "interface_backbone": {
                    "clamp_distance": 30.0,
                    "loss_unit_distance": 20.0,
                    "weight": 0.5,
                },
            },
            "masked_msa": {"num_classes": 22},
            "violation": {
                "average_clashes": True,
                "weight": 0.03,  # Not finetuning
            },
            "tm": {"weight": 0.1, "enabled": True},
            "chain_center_of_mass": {"weight": 0.05, "enabled": True},
        },
    }
)

config = update_config_dict(monomer_config, multimer_config_update)
