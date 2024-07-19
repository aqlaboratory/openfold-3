########################
# Primitives
########################

mha_init = {
    "linear_q": {"bias": False, "init": "glorot"},
    "linear_k": {"bias": False, "init": "glorot"},
    "linear_v": {"bias": False, "init": "glorot"},
    "linear_g": {"bias": True, "init": "gating"},
    "linear_o": {"bias": True, "init": "final"},
}

mha_bias_init = {
    "linear_z": {"bias": False, "init": "normal"},
    "mha": mha_init,
}

########################
# Layers
########################

tri_mul_init = {
    "linear_a_p": {"bias": True, "init": "default"},
    "linear_a_g": {"bias": True, "init": "gating"},
    "linear_b_p": {"bias": True, "init": "default"},
    "linear_b_g": {"bias": True, "init": "gating"},
    "linear_g": {"bias": True, "init": "gating"},
    "linear_z": {"bias": True, "init": "final"},
}

tri_att_init = mha_bias_init

opm_init = {
    "linear_1": {"bias": True, "init": "default"},
    "linear_2": {"bias": True, "init": "default"},
    "linear_out": {"bias": True, "init": "final"},
}

template_pointwise_init = {"mha": mha_init}

msa_global_att_init = {"mha": mha_init}

transition_init = {
    "layers": {"bias": True, "init": "relu"},
    "linear_out": {"bias": True, "init": "final"},
}

########################
# Feature Embedders
########################

input_emb_init = {
    "linear_tf_z_i": {"bias": True, "init": "default"},
    "linear_tf_z_j": {"bias": True, "init": "default"},
    "linear_tf_m": {"bias": True, "init": "default"},
    "linear_msa_m": {"bias": True, "init": "default"},
    "linear_relpos": {"bias": True, "init": "default"},
}

recycling_emb_init = {"linear": {"bias": True, "init": "default"}}

extra_msa_emb_init = {"linear": {"bias": True, "init": "default"}}

templ_single_feat_emb_init = {
    "linear_1": {"bias": True, "init": "relu"},
    "linear_2": {"bias": True, "init": "relu"},
}

templ_pair_feat_emb_init = {
    "linear": {"bias": True, "init": "relu"},
}

########################
# Heads
########################
# TODO: Add heads config

########################
# Latent
########################

pair_block_init = {
    "tri_mul": tri_mul_init,
    "tri_att": tri_att_init,
    "pair_transition": transition_init,
}

msa_block_init = {
    "msa_row_att": mha_bias_init,
    "msa_transition": transition_init,
    "opm": opm_init,
    "pair_block": pair_block_init,
}

evo_block_init = {
    **msa_block_init,
    "msa_col_att": mha_bias_init,
    "linear": {"bias": True, "init": "default"},
}

extra_msa_block_init = {**msa_block_init, "msa_col_att": msa_global_att_init}

########################
# Structure
########################

angle_resnet_block_init = {
    "linear_1": {"bias": True, "init": "default"},
    "linear_2": {"bias": True, "init": "final"},
}

angle_resnet_init = {
    "linear_in": {"bias": True, "init": "default"},
    "linear_initial": {"bias": True, "init": "default"},
    "angle_resnet_block": angle_resnet_block_init,
    "linear_out": {"bias": True, "init": "default"},
}

point_proj_init = {
    "linear": {"bias": True, "init": "default"},
}

ipa_init = {
    "linear_q": {"bias": True, "init": "default"},
    "linear_q_points": point_proj_init,
    "linear_kv": {"bias": True, "init": "default"},
    "linear_kv_points": point_proj_init,
    "linear_b": {"bias": True, "init": "default"},
    "linear_out": {"bias": True, "init": "final"},
}

bb_update_init = {
    "linear": {"bias": True, "init": "final"},
}

structure_module_init = {
    "linear_in": {"bias": True, "init": "default"},
    "ipa": ipa_init,
    "transition": transition_init,
    "bb_update": bb_update_init,
    "angle_resnet": angle_resnet_init,
}
