import ml_collections as mlc

from openfold3.projects.of3_all_atom.config import (
    linear_init_config as lin_init,
)

# Hidden dimensions
c_s = mlc.FieldReference(384, field_type=int)
c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(64, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_atom = mlc.FieldReference(128, field_type=int)
c_atom_pair = mlc.FieldReference(16, field_type=int)
c_token_embedder = mlc.FieldReference(384, field_type=int)
c_token_diffusion = mlc.FieldReference(768, field_type=int)
c_s_input = mlc.FieldReference(c_token_embedder + 65, field_type=int)

# Diffusion parameters
sigma_data = mlc.FieldReference(16, field_type=int)
max_relative_idx = mlc.FieldReference(32, field_type=int)
max_relative_chain = mlc.FieldReference(2, field_type=int)
diffusion_training_enabled = mlc.FieldReference(True, field_type=bool)
n_query = mlc.FieldReference(32, field_type=int)
n_key = mlc.FieldReference(128, field_type=int)

# templates_enabled = mlc.FieldReference(True, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
inf = mlc.FieldReference(1e9, field_type=float)
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
ckpt_intermediate_steps = mlc.FieldReference(False, field_type=bool)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)
max_atoms_per_token = mlc.FieldReference(23, field_type=int)

# Cutoffs for chunking ops per diffusion sample
per_sample_token_cutoff = mlc.FieldReference(750, field_type=int)
per_sample_atom_cutoff = mlc.FieldReference(10000, field_type=int)
low_mem_validation = mlc.FieldReference(False, field_type=bool)

model_selection_metric_weights_config = mlc.FrozenConfigDict(
    {
        "initial_training": {
            "lddt_intra_modified_residues": 10.0,
            "lddt_inter_ligand_rna": 5.0,
            "lddt_inter_ligand_dna": 5.0,
            "lddt_intra_protein": 20.0,
            "lddt_intra_ligand": 20.0,
            "lddt_intra_dna": 4.0,
            "lddt_intra_rna": 16.0,
            "lddt_inter_protein_protein": 20.0,
            "lddt_inter_protein_ligand": 10.0,
            "lddt_inter_protein_dna": 10.0,
            "lddt_inter_protein_rna": 10.0,
            "rasa": 10.0,
        },
        "fine_tuning": {
            "lddt_inter_ligand_rna": 2.0,
            "lddt_inter_ligand_dna": 5.0,
            "lddt_intra_protein": 20.0,
            "lddt_intra_ligand": 20.0,
            "lddt_intra_dna": 4.0,
            "lddt_intra_rna": 16.0,
            "lddt_inter_protein_protein": 20.0,
            "lddt_inter_protein_ligand": 10.0,
            "lddt_inter_protein_dna": 10.0,
            "lddt_inter_protein_rna": 2.0,
            "rasa": 10.0,
        },
    }
)

model_config = mlc.ConfigDict(
    {
        "settings": {
            "memory": {
                "train": {
                    "chunk_size": None,
                    # Use DeepSpeed memory-efficient attention kernel. Mutually
                    # exclusive with use_lma.
                    "use_deepspeed_evo_attention": True,
                    "use_cueq_triangle_kernels": False,
                    # Use Staats & Rabe's low-memory attention algorithm. Mutually
                    # exclusive with use_deepspeed_evo_attention.
                    "use_lma": False,
                    "msa_module": {
                        "swiglu_chunk_token_cutoff": None,
                        "swiglu_seq_chunk_size": None,
                    },
                },
                "eval": {
                    "chunk_size": None,
                    "use_deepspeed_evo_attention": True,
                    "use_cueq_triangle_kernels": False,
                    "use_lma": False,
                    "msa_module": {
                        "swiglu_chunk_token_cutoff": None,
                        "swiglu_seq_chunk_size": None,
                    },
                    "per_sample_token_cutoff": per_sample_token_cutoff,
                    "per_sample_atom_cutoff": per_sample_atom_cutoff,
                    "low_mem_validation": low_mem_validation,
                    "offload_inference": {
                        "msa_module": False,
                        "confidence_heads": False,
                        "token_cutoff": None,
                    },
                },
            },
            # TODO: Remove field ref, copy manually for global setting
            #  to allow per-module overrides
            "blocks_per_ckpt": blocks_per_ckpt,
            "ckpt_intermediate_steps": ckpt_intermediate_steps,
            "clear_cache_between_steps": False,
            "diffusion_training_enabled": diffusion_training_enabled,
            "optimizer": {
                "use_deepspeed_adam": False,
                "learning_rate": 1.8e-3,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
            },
            "lr_scheduler": {
                "base_lr": 0.0,
                "warmup_no_steps": 1000,
                "start_decay_after_n_steps": 50000,
                "decay_every_n_steps": 50000,
                "decay_factor": 0.95,
            },
            "ema": {"decay": 0.999},
            "gradient_clipping": 10.0,
            "model_selection_weight_scheme": "initial_training",
        },
        "architecture": {
            "shared": {
                "c_s_input": c_s_input,
                "c_s": c_s,
                "c_z": c_z,
                "num_recycles": 3,
                "diffusion": {
                    "sigma_data": sigma_data,
                    "no_samples": 48,
                    "no_mini_rollout_samples": 1,
                    "no_full_rollout_samples": 5,
                    "no_mini_rollout_steps": 20,
                    "no_full_rollout_steps": 200,
                },
            },
            "input_embedder": {
                "c_s_input": c_s_input,
                "c_s": c_s,
                "c_z": c_z,
                "max_relative_idx": max_relative_idx,
                "max_relative_chain": max_relative_chain,
                "atom_attn_enc": {
                    "c_s": c_s,
                    "c_z": c_z,
                    "c_atom_ref": {
                        "element": 119,
                        "name_chars": 256,
                    },
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
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "ckpt_intermediate_steps": ckpt_intermediate_steps,
                    "inf": inf,
                    "linear_init_params": lin_init.atom_att_enc_init,
                    "use_reentrant": False,
                },
                "linear_init_params": lin_init.input_emb_init,
            },
            "template": {
                "c_t": c_t,
                "c_z": c_z,
                "linear_init_param": lin_init.templ_module_init,
                "template_pair_embedder": {
                    "c_in": c_z,
                    "c_dgram": 39,
                    "c_aatype": 32,
                    "c_out": c_t,
                    "linear_init_params": lin_init.templ_pair_feat_emb_init,
                },
                "template_pair_stack": {
                    "c_t": c_t,
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
                    "inf": inf,
                    "linear_init_params": lin_init.pair_block_init,
                    "use_reentrant": False,
                    "tune_chunk_size": tune_chunk_size,
                },
            },
            "msa": {
                "msa_module_embedder": {
                    "c_m_feats": 34,
                    "c_m": c_m,
                    "c_s_input": c_s_input,
                    "subsample_main_msa": True,
                    "subsample_all_msa": False,
                    "min_subsampled_all_msa": 1024,
                    "max_subsampled_all_msa": 1024,
                    "linear_init_params": lin_init.msa_module_emb_init,
                },
                "msa_module": {
                    "c_m": c_m,
                    "c_z": c_z,
                    "c_hidden_msa_att": 8,  # 8 or 32, possible typo in SI
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
                    "inf": inf,
                    "eps": eps,
                    "linear_init_params": lin_init.msa_module_init,
                    "use_reentrant": False,
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
                "inf": inf,
                "linear_init_params": lin_init.pairformer_init,
                "use_reentrant": False,
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
                    "c_atom_ref": {
                        "element": 119,
                        "name_chars": 256,
                    },
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
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "ckpt_intermediate_steps": ckpt_intermediate_steps,
                    "inf": inf,
                    "linear_init_params": lin_init.atom_att_enc_init,
                    "use_reentrant": False,
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
                    "n_query": None,
                    "n_key": None,
                    "inf": inf,
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "linear_init_params": lin_init.diffusion_transformer_init,
                    "use_reentrant": False,
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
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "inf": inf,
                    "linear_init_params": lin_init.atom_att_dec_init,
                    "use_reentrant": False,
                },
            },
            "noise_schedule": {
                "sigma_data": sigma_data,
                "s_max": 160.0,
                "s_min": 4e-4,
                "p": 7,
            },
            "sample_diffusion": {
                "gamma_0": 0.8,
                "gamma_min": 1.0,
                "noise_scale": 1.003,
                "step_scale": 1.5,
            },
            "heads": {
                "max_atoms_per_token": max_atoms_per_token,
                "per_sample_token_cutoff": per_sample_token_cutoff,
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
                        "inf": inf,
                        "linear_init_params": lin_init.pairformer_init,
                        "use_reentrant": False,
                        "clear_cache_between_blocks": False,
                        "tune_chunk_size": tune_chunk_size,
                    },
                    "c_s_input": c_s_input,
                    "c_z": c_z,
                    "min_bin": 3.25,
                    "max_bin": 50.75,
                    "no_bin": 39,
                    "inf": inf,
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
                    "max_atoms_per_token": max_atoms_per_token,
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
                    "max_atoms_per_token": max_atoms_per_token,
                    "linear_init_params": lin_init.exp_res_all_atom_init,
                },
            },
            "loss_module": {
                "per_sample_atom_cutoff": per_sample_atom_cutoff,
                "low_mem_validation": low_mem_validation,
                "confidence_loss_names": [
                    "plddt",
                    "pde",
                    "experimentally_resolved",
                    "pae",
                ],
                "diffusion_loss_names": ["bond", "smooth_lddt", "mse"],
                # TODO: Factor out the number bins from each of these
                "confidence": {
                    "plddt": {
                        "no_bins": 50,
                        "bin_min": 0.0,
                        "bin_max": 1.0,
                    },
                    "pde": {
                        "no_bins": 64,
                        "bin_min": 0.0,
                        "bin_max": 32.0,
                    },
                    "experimentally_resolved": {
                        "no_bins": 2,
                    },
                    "pae": {
                        "angle_threshold": 25.0,
                        "no_bins": 64,
                        "bin_min": 0.0,
                        "bin_max": 32.0,
                    },
                    "per_sample_atom_cutoff": per_sample_atom_cutoff,
                    "eps": eps,
                    "inf": inf,
                },
                "diffusion": {
                    "sigma_data": sigma_data,
                    "dna_weight": 5.0,
                    "rna_weight": 5.0,
                    "ligand_weight": 10.0,
                    "eps": eps,
                    "chunk_size": None,
                },
                "distogram": {
                    "no_bins": 64,
                    "bin_min": 2.0,
                    "bin_max": 22.0,
                    "eps": eps,
                },
            },
        },
        "confidence": {
            "per_sample_atom_cutoff": per_sample_atom_cutoff,
            "low_mem_validation": low_mem_validation,
            "plddt": {
				"no_bins": 50,
                "bin_min": 0,
                "bin_max": 1,
			},
            "pde": {
                "bin_min": 0,
                "bin_max": 32,
                "no_bins": 64,
                "return_probs": False,
            },
            "pae": {
                "bin_min": 0,
                "bin_max": 32,
                "no_bins": 64,
                "return_probs": False,
            },
            "distogram": {
                "bin_min": 2,
                "bin_max": 22,
                "no_bins": 64,
                "return_contact_probs": False,
            },
            "ptm": {
                "bin_min": 0,
                "bin_max": 32,
                "no_bins": 64,
            },
            "sample_ranking": {
                "full_complex": {
                    "ptm_weight": 0.2,
                    "iptm_weight": 0.8,
                    "disorder_weight": 0.5,
                    "has_clash_weight": -100.0,
                    "disorder_threshold": 0.581,
                },
                "all_ipTM": {"enabled": True},
                "all_pTM": {"enabled": True},
                "pair_chunk": 32,
            },
            "clash": {
                "min_distance": 1.1,
                "clash_cutoff_num": 100,
                "clash_cutoff_ratio": 0.5,
            },
            "rasa": {
                "cutoff": 0.581,
            },
        },
    }
)
