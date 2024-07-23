import ml_collections as mlc

import openfold3.model_implementations.af3_all_atom.linear_init_config as lin_init

NUM_TOKENS = "num tokens placeholder"
NUM_ATOMS = "num atoms placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_TEMPLATES = "num templates placeholder"


# Hidden dimensions
c_s = mlc.FieldReference(384, field_type=int)
c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(64, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_atom_ref = mlc.FieldReference(389, field_type=int)
c_atom = mlc.FieldReference(128, field_type=int)
c_atom_pair = mlc.FieldReference(16, field_type=int)
c_token_embedder = mlc.FieldReference(384, field_type=int)
c_token_diffusion = mlc.FieldReference(768, field_type=int)
c_s_input = mlc.FieldReference(c_token_embedder + 65, field_type=int)

# Diffusion parameters
sigma_data = mlc.FieldReference(16, field_type=int)
max_relative_idx = mlc.FieldReference(32, field_type=int)
max_relative_chain = mlc.FieldReference(2, field_type=int)
no_samples = mlc.FieldReference(48, field_type=int)
no_rollout_steps = mlc.FieldReference(20, field_type=int)
n_query = mlc.FieldReference(32, field_type=int)
n_key = mlc.FieldReference(128, field_type=int)
use_block_sparse_attn = mlc.FieldReference(False, field_type=bool)
block_size = mlc.FieldReference(16, field_type=int)

# templates_enabled = mlc.FieldReference(True, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(None, field_type=int)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)


config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "feat": {
                    "residue_index": [NUM_TOKENS],
                    "token_index": [NUM_TOKENS],
                    "asym_id": [NUM_TOKENS],
                    "entity_id": [NUM_TOKENS],
                    "sym_id": [NUM_TOKENS],
                    "restype": [NUM_TOKENS, 32],
                    "is_protein": [NUM_TOKENS],
                    "is_rna": [NUM_TOKENS],
                    "is_dna": [NUM_TOKENS],
                    "is_ligand": [NUM_TOKENS],
                    "ref_pos": [NUM_ATOMS, 3],
                    "ref_mask": [NUM_ATOMS],
                    "ref_element": [NUM_ATOMS, 128],
                    "ref_charge": [NUM_ATOMS],
                    "ref_atom_name_chars": [NUM_ATOMS, 4, 64],
                    "ref_space_uid": [NUM_ATOMS],
                    "msa": [NUM_MSA_SEQ, NUM_TOKENS, 32],
                    "has_deletion": [NUM_MSA_SEQ, NUM_TOKENS],
                    "deletion_value": [NUM_MSA_SEQ, NUM_TOKENS],
                    "profile": [NUM_TOKENS, 32],
                    "deletion_mean": [NUM_TOKENS],
                    "template_restype": [NUM_TEMPLATES, NUM_TOKENS, 32],
                    "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_TOKENS],
                    "template_backbone_frame_mask": [NUM_TEMPLATES, NUM_TOKENS],
                    "template_distogram": [NUM_TEMPLATES, NUM_TOKENS, NUM_TOKENS, 39],
                    "template_unit_vector": [NUM_TEMPLATES, NUM_TOKENS, NUM_TOKENS, 3],
                    "token_bonds": [NUM_TOKENS, NUM_TOKENS],
                    # Features not included in AF3 docs
                    "atom_to_token_index": [NUM_ATOMS],
                    "token_mask": [NUM_TOKENS],
                    "msa_mask": [NUM_MSA_SEQ, NUM_TOKENS],
                    "num_main_msa_seqs": [],
                    "gt_atom_positions": [NUM_ATOMS, 3],
                    "gt_atom_mask": [NUM_ATOMS],
                }
            }
        },
        "globals": {
            "c_s_input": c_s_input,
            "c_s": c_s,
            "c_z": c_z,
            "sigma_data": sigma_data,
            "no_cycles": 4,
            "no_samples": no_samples,
            "no_rollout_steps": no_rollout_steps,
            "blocks_per_ckpt": blocks_per_ckpt,
            "chunk_size": chunk_size,
            # Use DeepSpeed memory-efficient attention kernel. Mutually
            # exclusive with use_lma and use_flash.
            "use_deepspeed_evo_attention": False,
            # Use Staats & Rabe's low-memory attention algorithm. Mutually
            # exclusive with use_deepspeed_evo_attention and use_flash.
            "use_lma": False,
            "offload_inference": False,
        },
        "model": {
            "input_embedder": {
                "c_s_input": c_s_input,
                "c_s": c_s,
                "c_z": c_z,
                "max_relative_idx": max_relative_idx,
                "max_relative_chain": max_relative_chain,
                "atom_attn_enc": {
                    "c_s": c_s,
                    "c_z": c_z,
                    "c_atom_ref": c_atom_ref,
                    "c_atom": c_atom,
                    "c_atom_pair": c_atom_pair,
                    "c_token": c_token_embedder,
                    # c_atom / no_heads
                    # built into the function (might get float depending on conf.)
                    "c_hidden": 32,
                    "no_heads": 4,
                    "no_blocks": 3,
                    "n_transition": 2,
                    "n_query": n_query,
                    "n_key": n_key,
                    "use_ada_layer_norm": True,
                    "use_block_sparse_attn": use_block_sparse_attn,
                    "block_size": block_size,
                    "inf": 1e10,  # global parameter?
                    "linear_init_params": lin_init.atom_att_enc_init,
                },
                "linear_init_params": lin_init.input_emb_init,
            },
            "template": {
                "c_t": c_t,
                "c_z": c_z,
                "linear_init_param": lin_init.templ_module_init,
                "template_pair_embedder": {
                    "c_in": 108,
                    "c_z": c_z,
                    "c_out": c_t,
                    "linear_init_params": lin_init.templ_pair_feat_emb_init,
                },
                "template_pair_stack": {
                    "c_t": c_t,
                    # TODO: Do we use pairformer attn params?
                    # DISCREPANCY: c_hidden_tri_att here is given in the supplement
                    # as 64. In the code, it's 16.
                    "c_hidden_tri_att": 16,
                    "c_hidden_tri_mul": 64,
                    "no_blocks": 2,
                    "no_heads": 4,
                    "transition_type": "swiglu",
                    "pair_transition_n": 2,
                    "dropout_rate": 0.25,
                    "tri_mul_first": True,
                    "fuse_projection_weights": False,
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "inf": 1e9,
                    "linear_init_params": lin_init.pair_block_init,
                    "tune_chunk_size": tune_chunk_size,
                },
            },
            "msa": {
                "msa_module_embedder": {
                    "c_m_feats": 34,
                    "c_m": c_m,
                    "c_s_input": c_s_input,
                    "linear_init_params": lin_init.msa_module_emb_init,
                },
                "msa_module": {
                    "c_m": c_m,
                    "c_z": c_z,
                    "c_hidden_msa_att": 32,
                    "c_hidden_opm": 32,
                    "c_hidden_mul": 128,
                    "c_hidden_pair_att": 32,
                    "no_heads_msa": 8,
                    "no_heads_pair": 4,
                    "no_blocks": 4,
                    "transition_type": "swiglu",
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "opm_first": True,
                    "fuse_projection_weights": False,
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "inf": 1e9,
                    "eps": eps,
                    "linear_init_params": lin_init.msa_module_init,
                    "clear_cache_between_blocks": False,
                    "tune_chunk_size": tune_chunk_size,
                },
            },
            "pairformer": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden_pair_bias": 24,  # c_s / no_heads_pair_bias
                "no_heads_pair_bias": 16,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "no_heads_pair": 4,
                "no_blocks": 48,
                "transition_type": "swiglu",
                "transition_n": 4,
                "pair_dropout": 0.25,
                "fuse_projection_weights": False,
                "blocks_per_ckpt": blocks_per_ckpt,
                "inf": 1e9,
                "linear_init_params": lin_init.pairformer_init,
                "clear_cache_between_blocks": False,
                "tune_chunk_size": tune_chunk_size,
            },
            "diffusion_module": {
                "diffusion_module": {
                    "c_s": c_s,
                    "c_token": c_token_diffusion,
                    "sigma_data": sigma_data,
                    "linear_init_params": lin_init.diffusion_module_init,
                },
                "diffusion_conditioning": {
                    "c_s_input": c_s_input,
                    "c_s": c_s,
                    "c_z": c_z,
                    "sigma_data": sigma_data,
                    "c_fourier_emb": 256,
                    "max_relative_idx": max_relative_idx,
                    "max_relative_chain": max_relative_chain,
                    "linear_init_params": lin_init.diffusion_cond_init,
                },
                "atom_attn_enc": {
                    "c_s": c_s,
                    "c_z": c_z,
                    "c_atom_ref": c_atom_ref,
                    "c_atom": c_atom,
                    "c_atom_pair": c_atom_pair,
                    "c_token": c_token_diffusion,
                    # c_atom / no_heads
                    # built into the function (might get float depending on conf.)
                    "c_hidden": 32,
                    "no_heads": 4,
                    "no_blocks": 3,
                    "n_transition": 2,
                    "n_query": n_query,
                    "n_key": n_key,
                    "use_ada_layer_norm": True,
                    "use_block_sparse_attn": use_block_sparse_attn,
                    "block_size": block_size,
                    "inf": 1e9,  # global parameter?
                    "linear_init_params": lin_init.atom_att_enc_init,
                },
                "diffusion_transformer": {
                    "c_a": c_token_diffusion,
                    "c_s": c_s,
                    "c_z": c_z,
                    "c_hidden": 48,  # c_token / no_heads
                    "no_heads": 16,
                    "no_blocks": 24,
                    "n_transition": 2,
                    "use_ada_layer_norm": True,
                    "use_block_sparse_attn": False,
                    "block_size": None,
                    "inf": 1e9,  # global parameter?
                    "linear_init_params": lin_init.diffusion_transformer_init,
                },
                "atom_attn_dec": {
                    "c_atom": c_atom,
                    "c_atom_pair": c_atom_pair,
                    "c_token": c_token_diffusion,
                    "c_hidden": 32,  # c_atom / no_heads
                    "no_heads": 4,
                    "no_blocks": 3,
                    "n_transition": 2,
                    "n_query": n_query,
                    "n_key": n_key,
                    "use_ada_layer_norm": True,
                    "use_block_sparse_attn": use_block_sparse_attn,
                    "block_size": block_size,
                    "inf": 1e9,  # global parameter?
                    "linear_init_params": lin_init.atom_att_dec_init,
                },
            },
            "sample_diffusion": {
                "gamma_0": 0.8,
                "gamma_min": 1.0,
                "noise_scale": 1.003,
                "step_scale": 1.5,
                "no_rollout_steps": no_rollout_steps,
                "sigma_data": sigma_data,
                "s_max": 160.0,
                "s_min": 4e-4,
                "p": 7,
            },
            "heads": {
                "pairformer_embedding": {
                    "pairformer": {
                        "c_s": c_s,
                        "c_z": c_z,
                        "c_hidden_pair_bias": 24,  # c_s / no_heads_pair_bias
                        "no_heads_pair_bias": 16,
                        "c_hidden_mul": 128,
                        "c_hidden_pair_att": 32,
                        "no_heads_pair": 4,
                        "no_blocks": 4,
                        "transition_type": "swiglu",
                        "transition_n": 4,
                        "pair_dropout": 0.25,
                        "fuse_projection_weights": False,
                        "blocks_per_ckpt": blocks_per_ckpt,
                        "inf": 1e9,
                        "linear_init_params": lin_init.pairformer_init,
                        "clear_cache_between_blocks": False,
                        "tune_chunk_size": tune_chunk_size,
                    },
                    "c_s": c_s,
                    "c_z": c_z,
                    "min_bin": 3.25,
                    "max_bin": 20.75,
                    "no_bin": 15,
                    "inf": 1e8,
                    "linear_init_params": lin_init.pairformer_head_init,
                },
                "pae": {
                    "c_z": c_z,
                    "c_out": 64,
                    "linear_init_params": lin_init.pae_init,
                    "enabled": False,
                },
                "pde": {
                    "c_z": c_z,
                    "c_out": 64,
                    "linear_init_params": lin_init.pde_init,
                },
                "lddt": {
                    "c_s": c_s,
                    "c_out": 50,
                    "linear_init_params": lin_init.lddt_init,
                },
                "distogram": {
                    "c_z": c_z,
                    "c_out": 64,
                    "linear_init_params": lin_init.distogram_init,
                    "enabled": True,
                },
                "experimentally_resolved": {
                    "c_s": c_s,
                    "c_out": 2,
                    "linear_init_params": lin_init.exp_res_all_atom_init,
                },
            },
        },
        "loss": {
            "diffusion": {
                "sigma_data": sigma_data,
                "alpha_bond": 0.0,  # varies based on training and finetuning
                "alpha_dna": 5.0,
                "alpha_rna": 5.0,
                "alpha_ligand": 10.0,
                "enabled": True,
            }
        },
    }
)

train_config_update = mlc.ConfigDict({"loss": {"diffusion": {"alpha_bond": 0.0}}})

finetune1_config_update = mlc.ConfigDict({"loss": {"diffusion": {"alpha_bond": 1.0}}})

finetune2_config_update = mlc.ConfigDict({"loss": {"diffusion": {"alpha_bond": 1.0}}})

finetune3_config_update = mlc.ConfigDict(
    {
        "model": {"heads": {"pae": {"enabled": True}, "distogram": {"enabled": False}}},
        "loss": {"diffusion": {"enabled": False}},
    }
)

eval_config_update = mlc.ConfigDict({"globals": {"no_rollout_steps": 200}})
