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

template_pointwise_init = {
    "mha": mha_init
}

msa_row_col_att_init = mha_bias_init

msa_global_att_init = {
    "mha": mha_init
}

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

preembed_init = {
    "linear_tf_m": {"bias": True, "init": "default"},
    "linear_preemb_m": {"bias": True, "init": "default"},
    "linear_preemb_z_i": {"bias": True, "init": "default"},
    "linear_preemb_z_j": {"bias": True, "init": "default"},
    "linear_relpos": {"bias": True, "init": "default"}
}

recycling_emb_init = {
    "linear": {"bias": True, "init": "default"}
}

extra_msa_emb_init = {
    "linear": {"bias": True, "init": "default"}
}

templ_single_feat_emb_init = {
    "linear_1": {"bias": True, "init": "relu"},
    "linear_2": {"bias": True, "init": "relu"}
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

msa_block_init = {

}

pair_block_init = {
    "tri_mul_init": tri_mul_init,
    "tri_att_init": tri_att_init,
    "transition_init": transition_init,
}

########################
# Structure
########################
