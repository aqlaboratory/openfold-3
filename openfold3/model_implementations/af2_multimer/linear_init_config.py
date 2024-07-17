default_bias_init = {"bias": True, "init": "default"}
final_bias_init = {"bias": True, "init": "final"}
gating_bias_init = {"bias": True, "init": "gating"}
glorot_bias_init = {"bias": False, "init": "glorot"}
normal_bias_init = {"bias": False, "init": "normal"}
relu_bias_init = {"bias": True, "init": "relu"}

########################
# Primitives
########################

mha_init = {
    "linear_q": glorot_bias_init,
    "linear_k": glorot_bias_init,
    "linear_v": glorot_bias_init,
    "linear_g": gating_bias_init,
    "linear_o": final_bias_init,
}

mha_bias_init = {
    "linear_z": normal_bias_init,
    "mha": mha_init,
}

########################
# Layers
########################

tri_mul_init = {
    "linear_ab_p": default_bias_init,
    "linear_ab_g": gating_bias_init,
    "linear_g": gating_bias_init,
    "linear_z": final_bias_init,
}

tri_att_init = mha_bias_init

opm_init = {
    "linear_1": default_bias_init,
    "linear_2": default_bias_init,
    "linear_out": final_bias_init,
}

msa_row_col_init = mha_bias_init

msa_global_init = mha_init

transition_init = {
    "layers": relu_bias_init,
    "linear_out": final_bias_init,
}

########################
# Feature Embedders
########################

input_emb_init = {}

templ_feat_emb_init = {}

########################
# Heads
########################
# TODO: Add heads config

########################
# Latent
########################

pair_block_init = {
    "tri_mul_init": tri_mul_init,
    "tri_att_init": tri_att_init,
    "transition_init": transition_init,
}

########################
# Structure
########################
