"""Linear layer initialization configuration for AlphaFold2 multimer model."""

from ml_collections import ConfigDict

########################
# Primitives
########################

swiglu_init = ConfigDict(
    {
        "linear_a": {"bias": False, "init": "he_normal"},
        "linear_b": {"bias": False, "init": "he_normal"},
    }
)

mha_init = ConfigDict(
    {
        "linear_q": {"bias": False, "init": "glorot"},
        "linear_k": {"bias": False, "init": "glorot"},
        "linear_v": {"bias": False, "init": "glorot"},
        "linear_g": {"bias": True, "init": "gating"},
        "linear_o": {"bias": True, "init": "final"},
    }
)

mha_bias_init = ConfigDict(
    {
        "linear_z": {"bias": False, "init": "normal"},
        "mha": mha_init,
    }
)

########################
# Layers
########################

# Not used by default, but config is included in case the
# "fuse_projection_weights" option is set to False
tri_mul_init = ConfigDict(
    {
        "linear_a_p": {"bias": True, "init": "default"},
        "linear_a_g": {"bias": True, "init": "gating"},
        "linear_b_p": {"bias": True, "init": "default"},
        "linear_b_g": {"bias": True, "init": "gating"},
        "linear_g": {"bias": True, "init": "gating"},
        "linear_z": {"bias": True, "init": "final"},
    }
)

fused_tri_mul_init = ConfigDict(
    {
        "linear_ab_p": {"bias": True, "init": "default"},
        "linear_ab_g": {"bias": True, "init": "gating"},
        "linear_g": {"bias": True, "init": "gating"},
        "linear_z": {"bias": True, "init": "final"},
    }
)

tri_att_init = mha_bias_init

opm_init = ConfigDict(
    {
        "linear_1": {"bias": True, "init": "default"},
        "linear_2": {"bias": True, "init": "default"},
        "linear_out": {"bias": True, "init": "final"},
    }
)

msa_global_att_init = ConfigDict({"mha": mha_init})

swiglu_transition_init = ConfigDict(
    {
        "swiglu": swiglu_init,
        "linear_out": {"bias": False, "init": "final"},
    }
)

relu_transition_init = ConfigDict(
    {
        "layers": {"bias": True, "init": "relu"},
        "linear_out": {"bias": True, "init": "final"},
    }
)

transition_init = {"swiglu": swiglu_transition_init, "relu": relu_transition_init}

########################
# Feature Embedders
########################

input_emb_init = ConfigDict(
    {
        "linear_tf_z_i": {"bias": True, "init": "default"},
        "linear_tf_z_j": {"bias": True, "init": "default"},
        "linear_tf_m": {"bias": True, "init": "default"},
        "linear_msa_m": {"bias": True, "init": "default"},
        "linear_relpos": {"bias": True, "init": "default"},
    }
)

recycling_emb_init = ConfigDict({"linear": {"bias": True, "init": "default"}})

extra_msa_emb_init = ConfigDict({"linear": {"bias": True, "init": "default"}})

templ_single_feat_emb_init = ConfigDict(
    {
        "template_single_embedder": {"bias": True, "init": "default"},
        "template_projector": {"bias": True, "init": "default"},
    }
)

templ_pair_feat_emb_init = ConfigDict(
    {
        "dgram_linear": {"bias": True, "init": "relu"},
        "aatype_linear_1": {"bias": True, "init": "relu"},
        "aatype_linear_2": {"bias": True, "init": "relu"},
        "query_embedding_linear": {"bias": True, "init": "relu"},
        "pseudo_beta_mask_linear": {"bias": True, "init": "relu"},
        "x_linear": {"bias": True, "init": "relu"},
        "y_linear": {"bias": True, "init": "relu"},
        "z_linear": {"bias": True, "init": "relu"},
        "backbone_mask_linear": {"bias": True, "init": "relu"},
    }
)

########################
# Heads
########################

lddt_ca_init = ConfigDict(
    {
        "linear_1": {"bias": True, "init": "relu"},
        "linear_2": {"bias": True, "init": "relu"},
        "linear_3": {"bias": True, "init": "final"},
    }
)

exp_res_init = ConfigDict({"linear": {"bias": True, "init": "final"}})

distogram_init = ConfigDict({"linear": {"bias": True, "init": "final"}})

tm_score_init = ConfigDict({"linear": {"bias": True, "init": "final"}})

masked_msa_init = ConfigDict({"linear": {"bias": True, "init": "final"}})

########################
# Latent
########################

pair_block_init = ConfigDict(
    {
        "tri_mul": tri_mul_init,
        "fused_tri_mul": fused_tri_mul_init,
        "tri_att": tri_att_init,
        "pair_transition": transition_init,
    }
)

msa_block_init = ConfigDict(
    {
        "msa_row_att": mha_bias_init,
        "msa_transition": transition_init,
        "opm": opm_init,
        "pair_block": pair_block_init,
    }
)

evo_block_init = ConfigDict(
    {
        **msa_block_init,
        "msa_col_att": mha_bias_init,
        "linear": {"bias": True, "init": "default"},
    }
)

extra_msa_block_init = ConfigDict(
    {**msa_block_init, "msa_col_att": msa_global_att_init}
)

templ_module_init = ConfigDict({"linear_t": {"bias": True, "init": "default"}})

########################
# Structure
########################

angle_resnet_block_init = ConfigDict(
    {
        "linear_1": {"bias": True, "init": "default"},
        "linear_2": {"bias": True, "init": "final"},
    }
)

angle_resnet_init = ConfigDict(
    {
        "linear_in": {"bias": True, "init": "default"},
        "linear_initial": {"bias": True, "init": "default"},
        "angle_resnet_block": angle_resnet_block_init,
        "linear_out": {"bias": True, "init": "default"},
    }
)

point_proj_init = ConfigDict(
    {
        "linear": {"bias": True, "init": "default"},
    }
)

ipa_init = ConfigDict(
    {
        "linear_q": {"bias": False, "init": "default"},
        "linear_q_points": point_proj_init,
        "linear_k": {"bias": False, "init": "default"},
        "linear_v": {"bias": False, "init": "default"},
        "linear_k_points": point_proj_init,
        "linear_v_points": point_proj_init,
        "linear_b": {"bias": True, "init": "default"},
        "linear_out": {"bias": True, "init": "final"},
    }
)

bb_update_init = ConfigDict(
    {
        "linear": {"bias": True, "init": "final"},
    }
)

structure_module_init = ConfigDict(
    {
        "linear_in": {"bias": True, "init": "default"},
        "ipa": ipa_init,
        "transition": relu_transition_init,
        "bb_update": bb_update_init,
        "angle_resnet": angle_resnet_init,
    }
)
