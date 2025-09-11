"""
Defaults for Linear layer weight and bias initializations. Common modules
(i.e. Triangle Attention, OPM) default to the AF3 settings. These defaults
exist to allow users to import the individual modules without having to
specifying the initialization settings.
"""

from ml_collections import ConfigDict

########################
# Primitives
########################

# AF3
swiglu_init = ConfigDict(
    {
        "linear_a": {"bias": False, "init": "relu"},
        "linear_b": {"bias": False, "init": "relu"},
    }
)

# AF3
ada_ln_init = ConfigDict(
    {
        "linear_g": {"bias": True, "init": "final"},
        "linear_s": {"bias": False, "init": "final"},
    }
)

# AF3
mha_init = ConfigDict(
    {
        "linear_q": {"bias": False, "init": "default"},
        "linear_k": {"bias": False, "init": "default"},
        "linear_v": {"bias": False, "init": "default"},
        "linear_g": {"bias": False, "init": "gating"},
        "linear_o": {"bias": False, "init": "final"},
    }
)

# AF3
att_pair_bias_mha_init = ConfigDict(
    {
        "linear_q": {"bias": True, "init": "default"},
        "linear_k": {"bias": False, "init": "default"},
        "linear_v": {"bias": False, "init": "default"},
        "linear_g": {"bias": False, "init": "gating"},
        "linear_o": {"bias": False, "init": "final"},
    }
)

# AF3
att_pair_bias_mha_ada_init = ConfigDict(
    {
        "linear_q": {"bias": True, "init": "default"},
        "linear_k": {"bias": False, "init": "default"},
        "linear_v": {"bias": False, "init": "default"},
        "linear_g": {"bias": False, "init": "gating"},
        "linear_o": {"bias": False, "init": "default"},
    }
)

# AF3
mha_bias_init = ConfigDict(
    {
        "linear_z": {"bias": False, "init": "default"},
        "mha": mha_init,
    }
)

########################
# Layers
########################

# AF3
att_pair_bias_init = ConfigDict(
    {
        "linear_z": {"bias": False, "init": "default"},
        "layer_norm_z": {"create_offset": True},
        "mha": att_pair_bias_mha_init,
    }
)

# AF3
diffusion_att_pair_bias_init = ConfigDict(
    {
        "ada_ln": ada_ln_init,
        "linear_ada_out": {"bias": True, "init": "gating_ada_zero"},
        "linear_z": {"bias": False, "init": "default"},
        "layer_norm_z": {"create_offset": False},
        "mha": att_pair_bias_mha_ada_init,
    }
)

# AF3
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

# AF2-Multimer
fused_tri_mul_init = ConfigDict(
    {
        "linear_ab_p": {"bias": True, "init": "default"},
        "linear_ab_g": {"bias": True, "init": "gating"},
        "linear_g": {"bias": True, "init": "gating"},
        "linear_z": {"bias": True, "init": "final"},
    }
)

# AF3
tri_att_init = mha_bias_init

# AF3
opm_init = ConfigDict(
    {
        "linear_1": {"bias": False, "init": "default"},
        "linear_2": {"bias": False, "init": "default"},
        "linear_out": {"bias": True, "init": "final"},
    }
)

# AF2-Monomer
template_pointwise_init = ConfigDict({"mha": mha_init})

# AF2
msa_global_att_init = ConfigDict({"mha": mha_init})

# AF3
msa_pair_avg_init = ConfigDict(
    {
        "linear_z": {"bias": False, "init": "default"},
        "linear_v": {"bias": False, "init": "default"},
        "linear_g": {"bias": False, "init": "gating"},
        "linear_o": {"bias": False, "init": "final"},
    }
)

# AF3
swiglu_transition_init = ConfigDict(
    {
        "swiglu": swiglu_init,
        "linear_out": {"bias": False, "init": "final"},
    }
)

# AF2
relu_transition_init = ConfigDict(
    {
        "layers": {"bias": True, "init": "relu"},
        "linear_out": {"bias": True, "init": "final"},
    }
)

transition_init = {"swiglu": swiglu_transition_init, "relu": relu_transition_init}

# AF3
cond_transition_init = ConfigDict(
    {
        "ada_ln": ada_ln_init,
        "swiglu": swiglu_init,
        "linear_g": {"bias": True, "init": "gating_ada_zero"},
        "linear_out": {"bias": False, "init": "default"},
    }
)

# AF3
diffusion_transformer_init = ConfigDict(
    {
        "att_pair_bias": diffusion_att_pair_bias_init,
        "cond_transition": cond_transition_init,
    }
)

# AF3
ref_atom_emb_init = ConfigDict(
    {
        "linear_feats": {"bias": False, "init": "default"},
        "linear_ref_offset": {"bias": False, "init": "default"},
        "linear_inv_sq_dists": {"bias": False, "init": "default"},
        "linear_valid_mask": {"bias": False, "init": "default"},
    }
)

# AF3
noisy_pos_emb_init = ConfigDict(
    {
        "linear_s": {"bias": False, "init": "final"},
        "linear_z": {"bias": False, "init": "final"},
        "linear_r": {"bias": False, "init": "default"},
    }
)

# AF3
atom_att_enc_init = ConfigDict(
    {
        "ref_atom_emb": ref_atom_emb_init,
        "noisy_pos_emb": noisy_pos_emb_init,
        "linear_l": {"bias": False, "init": "default"},
        "linear_m": {"bias": False, "init": "default"},
        "pair_mlp_1": {"bias": False, "init": "relu"},
        "pair_mlp_2": {"bias": False, "init": "relu"},
        "pair_mlp_3": {"bias": False, "init": "final"},
        "diffusion_transformer": diffusion_transformer_init,
        "linear_q": {"bias": False, "init": "default"},
    }
)

# AF3
atom_att_dec_init = ConfigDict(
    {
        "linear_q_in": {"bias": False, "init": "default"},
        "diffusion_transformer": diffusion_transformer_init,
        "linear_q_out": {"bias": False, "init": "final"},
    }
)

# AF3
diffusion_cond_init = ConfigDict(
    {
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

all_atom_input_emb_init = ConfigDict(
    {
        "linear_s": {"bias": False, "init": "default"},
        "linear_z_i": {"bias": False, "init": "default"},
        "linear_z_j": {"bias": False, "init": "default"},
        "linear_relpos": {"bias": False, "init": "default"},
        "linear_token_bonds": {"bias": False, "init": "default"},
    }
)

# AF3
msa_module_emb_init = ConfigDict(
    {
        "linear_m": {"bias": False, "init": "default"},
        "linear_s_input": {"bias": False, "init": "default"},
    }
)

# AF2-Monomer
monomer_templ_single_feat_emb_init = ConfigDict(
    {
        "linear_1": {"bias": True, "init": "relu"},
        "linear_2": {"bias": True, "init": "relu"},
    }
)

# AF2-Monomer
monomer_templ_pair_feat_emb_init = ConfigDict(
    {
        "linear": {"bias": True, "init": "relu"},
    }
)

# AF2-Multimer
multimer_templ_single_feat_emb_init = ConfigDict(
    {
        "template_single_embedder": {"bias": True, "init": "default"},
        "template_projector": {"bias": True, "init": "default"},
    }
)

# AF2-Multimer
multimer_templ_pair_feat_emb_init = ConfigDict(
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

# AF3
all_atom_templ_pair_feat_emb_init = ConfigDict(
    {
        "linear_a": {"bias": False, "init": "default"},
        "linear_z": {"bias": False, "init": "relu"},
    }
)

########################
# Heads
########################

# AF2
lddt_ca_init = ConfigDict(
    {
        "linear_1": {"bias": True, "init": "relu"},
        "linear_2": {"bias": True, "init": "relu"},
        "linear_3": {"bias": True, "init": "final"},
    }
)

# AF2
exp_res_init = ConfigDict({"linear": {"bias": True, "init": "final"}})

# AF2
tm_score_init = ConfigDict({"linear": {"bias": True, "init": "final"}})

# AF2
masked_msa_init = ConfigDict({"linear": {"bias": True, "init": "final"}})

# AF3
pairformer_head_init = ConfigDict(
    {
        "linear_i": {"bias": False, "init": "default"},
        "linear_j": {"bias": False, "init": "default"},
        "linear_distance": {"bias": False, "init": "default"},
    }
)

# AF3
pae_init = ConfigDict({"linear": {"bias": False, "init": "final"}})

# AF3
pde_init = ConfigDict({"linear": {"bias": False, "init": "final"}})

# AF3
lddt_init = ConfigDict({"linear": {"bias": False, "init": "final"}})

# AF3
exp_res_all_atom_init = ConfigDict({"linear": {"bias": False, "init": "final"}})

# AF3
distogram_init = ConfigDict({"linear": {"bias": False, "init": "final"}})

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

# AF2
msa_block_init = ConfigDict(
    {
        "msa_row_att": mha_bias_init,
        "msa_transition": transition_init,
        "opm": opm_init,
        "pair_block": pair_block_init,
    }
)

# AF2
evo_block_init = ConfigDict(
    {
        **msa_block_init,
        "msa_col_att": mha_bias_init,
        "linear": {"bias": True, "init": "default"},
    }
)

# AF2
extra_msa_block_init = ConfigDict(
    {**msa_block_init, "msa_col_att": msa_global_att_init}
)

# AF3
msa_module_init = ConfigDict(
    {
        **msa_block_init,
        "msa_pair_avg": msa_pair_avg_init,
    }
)

# AF3
pairformer_init = ConfigDict(
    {
        "pair_block": pair_block_init,
        "att_pair_bias": att_pair_bias_init,
        "transition": swiglu_transition_init,
    }
)

# AF2-Multimer
multimer_templ_module_init = ConfigDict({"linear_t": {"bias": True, "init": "default"}})

# AF3
all_atom_templ_module_init = ConfigDict(
    {"linear_t": {"bias": False, "init": "default"}}
)

########################
# Structure
########################

# AF2
angle_resnet_block_init = ConfigDict(
    {
        "linear_1": {"bias": True, "init": "default"},
        "linear_2": {"bias": True, "init": "final"},
    }
)

# AF2
angle_resnet_init = ConfigDict(
    {
        "linear_in": {"bias": True, "init": "default"},
        "linear_initial": {"bias": True, "init": "default"},
        "angle_resnet_block": angle_resnet_block_init,
        "linear_out": {"bias": True, "init": "default"},
    }
)

# AF2
point_proj_init = ConfigDict(
    {
        "linear": {"bias": True, "init": "default"},
    }
)

# AF2-Monomer
monomer_ipa_init = ConfigDict(
    {
        "linear_q": {"bias": True, "init": "default"},
        "linear_q_points": point_proj_init,
        "linear_kv": {"bias": True, "init": "default"},
        "linear_kv_points": point_proj_init,
        "linear_b": {"bias": True, "init": "default"},
        "linear_out": {"bias": True, "init": "final"},
    }
)

# AF2-Multimer
multimer_ipa_init = ConfigDict(
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

# AF2
bb_update_init = ConfigDict(
    {
        "linear": {"bias": True, "init": "final"},
    }
)

# AF2-Monomer
monomer_structure_module_init = ConfigDict(
    {
        "linear_in": {"bias": True, "init": "default"},
        "ipa": monomer_ipa_init,
        "transition": relu_transition_init,
        "bb_update": bb_update_init,
        "angle_resnet": angle_resnet_init,
    }
)

# AF2-Multimer
multimer_structure_module_init = ConfigDict(
    {
        "linear_in": {"bias": True, "init": "default"},
        "ipa": multimer_ipa_init,
        "transition": relu_transition_init,
        "bb_update": bb_update_init,
        "angle_resnet": angle_resnet_init,
    }
)

# AF3
diffusion_module_init = ConfigDict(
    {
        "linear_s": {"bias": False, "init": "final"},
    }
)
