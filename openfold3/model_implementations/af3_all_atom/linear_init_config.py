default_bias_init = {"bias": False, "init": "default"}
gating_bias_init = {"bias": True, "init": "gating"}
glorot_bias_init = {"bias": False, "init": "glorot"}
final_bias_init = {"bias": False, "init": "final"}  # Same as gating init with no bias
normal_bias_init = {"bias": False, "init": "normal"}
relu_bias_init = {"bias": False, "init": "relu"}

########################
# Primitives
########################

swiglu_init = {
    "linear_a": {"bias": False, "init": "he_normal"},
    "linear_b": {"bias": False, "init": "he_normal"},
}

# TODO: Verify these inits
ada_ln_init = {"linear_g": gating_bias_init, "linear_s": final_bias_init}

mha_init = {
    "linear_q": glorot_bias_init,
    "linear_k": glorot_bias_init,
    "linear_v": glorot_bias_init,
    "linear_g": final_bias_init,
    "linear_o": final_bias_init,
}

########################
# Layers
########################

att_pair_bias_init = {
    "ada_ln": ada_ln_init,
    "linear_ada_out": {"bias": True, "init": "gating_ada_zero"},
    "linear_z": normal_bias_init,
    "mha": {
        "linear_q": {"bias": True, "init": "glorot"},
        "linear_k": glorot_bias_init,
        "linear_v": glorot_bias_init,
        "linear_g": final_bias_init,
        "linear_o": final_bias_init,
    },
}

tri_mul_init = {
    "linear_a_p": default_bias_init,
    "linear_a_g": final_bias_init,
    "linear_b_p": default_bias_init,
    "linear_b_g": final_bias_init,
    "linear_g": final_bias_init,
    "linear_z": final_bias_init,
}

tri_att_init = {
    "linear_z": normal_bias_init,
    "mha": mha_init,
}

opm_init = {
    "linear_1": default_bias_init,
    "linear_2": default_bias_init,
    "linear_out": {"bias": True, "init": "final"},
}

msa_pair_bias_init = {
    "linear_z": normal_bias_init,
    "linear_v": glorot_bias_init,
    "linear_o": final_bias_init,
    "linear_g": gating_bias_init,
}

transition_init = {
    "swiglu": swiglu_init,
    "linear_out": final_bias_init,
}

cond_transition_init = {
    "ada_ln": ada_ln_init,
    "swiglu": swiglu_init,
    "linear_g": {"bias": True, "init": "gating_ada_zero"},
    "linear_out": final_bias_init,
}

diffusion_transformer_init = {
    "att_pair_bias": att_pair_bias_init,
    "cond_transition": cond_transition_init,
}

atom_transformer_init = {"diffusion_transformer": diffusion_transformer_init}

ref_atom_emb_init = {
    "linear_feats": default_bias_init,
    "linear_ref_offset": default_bias_init,
    "linear_inv_sq_dists": default_bias_init,
    "linear_valid_mask": default_bias_init,
}

noisy_pos_emb_init = {
    "linear_s": default_bias_init,
    "linear_z": default_bias_init,
    "linear_r": default_bias_init,
}

atom_att_enc_init = {
    "ref_atom_emb": ref_atom_emb_init,
    "noisy_pos_emb": noisy_pos_emb_init,
    "linear_l": relu_bias_init,
    "linear_m": relu_bias_init,
    "pair_mlp": relu_bias_init,
    "atom_transformer": atom_transformer_init,
    "linear_q": relu_bias_init,
}

atom_att_dec_init = {
    "linear_q_in": default_bias_init,
    "atom_transformer": atom_transformer_init,
    "linear_q_out": final_bias_init,
}

diffusion_conditioning = {
    "linear_z": default_bias_init,
    "transition_z": transition_init,
    "linear_s": default_bias_init,
    "fourier_emb": {"linear": {"bias": True, "init": "fourier"}},
    "linear_n": default_bias_init,
    "transition_s": transition_init,
}

########################
# Feature Embedders
########################

input_emb_init = {
    "linear_s": default_bias_init,
    "linear_z_i": default_bias_init,
    "linear_z_j": default_bias_init,
    "atom_att_enc": {},
}

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

msa_module_emb_init = {}

msa_module_init = {
    "pair_block_init": pair_block_init,
    "opm_init": opm_init,
}

pairformer_init = {}

########################
# Structure
########################
