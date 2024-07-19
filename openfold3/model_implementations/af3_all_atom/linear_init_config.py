"""Linear layer initialization configuration for AlphaFold3 model."""

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

# TODO: Verify these inits
ada_ln_init = ConfigDict(
    {
        "linear_g": {"bias": True, "init": "gating"},
        "linear_s": {"bias": False, "init": "final"},
    }
)

mha_init = ConfigDict(
    {
        "linear_q": {"bias": False, "init": "glorot"},
        "linear_k": {"bias": False, "init": "glorot"},
        "linear_v": {"bias": False, "init": "glorot"},
        "linear_g": {"bias": False, "init": "final"},
        "linear_o": {"bias": False, "init": "final"},
    }
)

block_sparse_mha_init = ConfigDict(
    {
        "linear_q": {"bias": True, "init": "glorot"},
        "linear_k": {"bias": False, "init": "glorot"},
        "linear_v": {"bias": False, "init": "glorot"},
        "linear_g": {"bias": False, "init": "final"},
        "linear_o": {"bias": False, "init": "final"},
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

att_pair_bias_init = ConfigDict(
    {
        "ada_ln": ada_ln_init,
        "linear_ada_out": {"bias": True, "init": "gating_ada_zero"},
        "linear_z": {"bias": False, "init": "normal"},
        "mha": block_sparse_mha_init,
    }
)

tri_mul_init = ConfigDict(
    {
        "linear_a_p": {"bias": False, "init": "default"},
        "linear_a_g": {"bias": False, "init": "final"},
        "linear_b_p": {"bias": False, "init": "default"},
        "linear_b_g": {"bias": False, "init": "final"},
        "linear_g": {"bias": False, "init": "final"},
        "linear_z": {"bias": False, "init": "final"},
    }
)

tri_att_init = mha_bias_init

opm_init = ConfigDict(
    {
        "linear_1": {"bias": False, "init": "default"},
        "linear_2": {"bias": False, "init": "default"},
        "linear_out": {"bias": True, "init": "final"},
    }
)

msa_pair_avg_init = ConfigDict(
    {
        "linear_z": {"bias": False, "init": "normal"},
        "linear_v": {"bias": False, "init": "glorot"},
        "linear_o": {"bias": False, "init": "final"},
        "linear_g": {"bias": False, "init": "gating"},
    }
)

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

cond_transition_init = ConfigDict(
    {
        "ada_ln": ada_ln_init,
        "swiglu": swiglu_init,
        "linear_g": {"bias": True, "init": "gating_ada_zero"},
        "linear_out": {"bias": False, "init": "final"},
    }
)

diffusion_transformer_init = ConfigDict(
    {
        "att_pair_bias": att_pair_bias_init,
        "cond_transition": cond_transition_init,
    }
)

atom_transformer_init = ConfigDict(
    {"diffusion_transformer": diffusion_transformer_init}
)

ref_atom_emb_init = ConfigDict(
    {
        "linear_feats": {"bias": False, "init": "default"},
        "linear_ref_offset": {"bias": False, "init": "default"},
        "linear_inv_sq_dists": {"bias": False, "init": "default"},
        "linear_valid_mask": {"bias": False, "init": "default"},
    }
)

noisy_pos_emb_init = ConfigDict(
    {
        "linear_s": {"bias": False, "init": "default"},
        "linear_z": {"bias": False, "init": "default"},
        "linear_r": {"bias": False, "init": "default"},
    }
)

atom_att_enc_init = ConfigDict(
    {
        "ref_atom_emb": ref_atom_emb_init,
        "noisy_pos_emb": noisy_pos_emb_init,
        "linear_l": {"bias": False, "init": "relu"},
        "linear_m": {"bias": False, "init": "relu"},
        "pair_mlp": {"bias": False, "init": "relu"},
        "atom_transformer": atom_transformer_init,
        "linear_q": {"bias": False, "init": "relu"},
    }
)

atom_att_dec_init = ConfigDict(
    {
        "linear_q_in": {"bias": False, "init": "default"},
        "atom_transformer": atom_transformer_init,
        "linear_q_out": {"bias": False, "init": "final"},
    }
)

relpos_emb_init = ConfigDict({"linear_relpos": {"bias": False, "init": "default"}})

diffusion_cond_init = ConfigDict(
    {
        "relpos_emb": relpos_emb_init,
        "linear_z": {"bias": False, "init": "default"},
        "transition_z": swiglu_transition_init,
        "linear_s": {"bias": False, "init": "default"},
        "linear_n": {"bias": False, "init": "default"},
        "transition_s": swiglu_transition_init,
    }
)

########################
# Feature Embedders
########################

input_emb_init = ConfigDict(
    {
        "linear_s": {"bias": False, "init": "default"},
        "linear_z_i": {"bias": False, "init": "default"},
        "linear_z_j": {"bias": False, "init": "default"},
        "relpos_emb": relpos_emb_init,
        "linear_token_bonds": {"bias": False, "init": "default"},
    }
)

msa_module_emb_init = ConfigDict(
    {
        "linear_m": {"bias": False, "init": "default"},
        "linear_s_input": {"bias": False, "init": "default"},
    }
)

# TODO: check initialization
templ_pair_feat_emb_init = ConfigDict(
    {
        "linear_a": {"bias": False, "init": "default"},
        "linear_z": {"bias": False, "init": "relu"},
    }
)

########################
# Heads
########################
# TODO: Add heads config

########################
# Latent
########################

pair_block_init = ConfigDict(
    {
        "tri_mul": tri_mul_init,
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

msa_module_init = ConfigDict(
    {
        **msa_block_init,
        "msa_pair_avg": msa_pair_avg_init,
    }
)

pairformer_init = ConfigDict(
    {
        "pair_block": pair_block_init,
        "att_pair_bias": att_pair_bias_init,
        "transition": swiglu_transition_init,
    }
)

templ_module_init = ConfigDict({"linear_t": {"bias": False, "init": "default"}})

########################
# Structure
########################

# TODO: Maybe structure this like other modules, where the configs are contained
# in one dict for the full module. Because of the way the params are passed,
# we only need to define the top level linear layers here.
diffusion_module_init = ConfigDict(
    {
        "linear_s": {"bias": False, "init": "default"},
    }
)
