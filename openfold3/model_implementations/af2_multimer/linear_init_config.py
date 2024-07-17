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
    "linear_ab_p": {"bias": True, "init": "default"},
    "linear_ab_g": {"bias": True, "init": "gating"},
    "linear_g": {"bias": True, "init": "gating"},
    "linear_z": {"bias": True, "init": "final"},
}

tri_att_init = mha_bias_init

opm_init = {
    "linear_1": {"bias": True, "init": "default"},
    "linear_2": {"bias": True, "init": "default"},
    "linear_out": {"bias": True, "init": "final"},
}

msa_row_col_att_init = mha_bias_init

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

templ_single_feat_emb_init = {
    "template_single_embedder": {"bias": True, "init": "default"},
    "template_projector": {"bias": True, "init": "default"},
}

templ_pair_feat_emb_init = {
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
