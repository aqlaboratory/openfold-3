from pathlib import Path

import ml_collections as mlc

from openfold3.projects.af3_all_atom.config import (
    linear_init_config as lin_init,
)

PLACEHOLDER_PATH = Path("placeholder/path")

# Hidden dimensions
c_s = mlc.FieldReference(384, field_type=int)
c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(64, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_atom_ref = mlc.FieldReference(379, field_type=int)
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
diffusion_training_enabled = mlc.FieldReference(True, field_type=bool)
n_query = mlc.FieldReference(32, field_type=int)
n_key = mlc.FieldReference(128, field_type=int)
use_block_sparse_attn = mlc.FieldReference(False, field_type=bool)
block_size = mlc.FieldReference(16, field_type=int)

# templates_enabled = mlc.FieldReference(True, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(None, field_type=int)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)
max_atoms_per_token = mlc.FieldReference(23, field_type=int)


project_config = mlc.ConfigDict(
    {
        "model": {
            "settings": {
                "blocks_per_ckpt": blocks_per_ckpt,
                "chunk_size": chunk_size,
                "use_block_sparse_attn": use_block_sparse_attn,
                "diffusion_training_enabled": diffusion_training_enabled,
                # Use DeepSpeed memory-efficient attention kernel. Mutually
                # exclusive with use_lma and use_flash.
                "use_deepspeed_evo_attention": False,
                # Use Staats & Rabe's low-memory attention algorithm. Mutually
                # exclusive with use_deepspeed_evo_attention and use_flash.
                "use_lma": False,
                "offload_inference": False,
                "optimizer": {
                    "learning_rate": 1.8e-3,
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eps": 1e-8,
                },
                "ema": {"decay": 0.999},
                "gradient_clipping": 10.0,
            },
            "architecture": {
                "shared": {
                    "c_s_input": c_s_input,
                    "c_s": c_s,
                    "c_z": c_z,
                    "no_cycles": 4,
                    "last_recycle_grad_only": True,
                    "diffusion": {
                        "sigma_data": sigma_data,
                        "no_samples": no_samples,
                        "no_rollout_steps": no_rollout_steps,
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
                        "use_reentrant": False,
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
                        "inf": 1e9,
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
                    "inf": 1e9,
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
                        "use_block_sparse_attn": use_block_sparse_attn,
                        "block_size": block_size,
                        "blocks_per_ckpt": blocks_per_ckpt,
                        "inf": 1e9,  # global parameter?
                        "linear_init_params": lin_init.atom_att_dec_init,
                        "use_reentrant": False,
                    },
                },
                "noise_schedule": {
                    "no_rollout_steps": no_rollout_steps,
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
                            "use_reentrant": False,
                            "clear_cache_between_blocks": False,
                            "tune_chunk_size": tune_chunk_size,
                        },
                        "c_s_input": c_s_input,
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
                    "confidence": {
                        "pde": {
                            "max_bin": 31,
                            "no_bins": 64,
                        },
                        "pae": {
                            "max_bin": 31,
                            "no_bins": 64,
                        },
                        "ptm": {
                            "max_bin": 31,
                            "no_bins": 64,
                            "ptm_weight": 0.2,
                            "iptm_weight": 0.8,
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
                },
                "loss_module": {
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
                            "weight": 0.0,
                        },
                        "eps": eps,
                        "inf": 1e9,
                    },
                    "diffusion": {
                        "sigma_data": sigma_data,
                        "dna_weight": 5.0,
                        "rna_weight": 5.0,
                        "ligand_weight": 10.0,
                        "eps": eps,
                        "chunk_size": None,  # 16 for 40GB GPUs
                    },
                    "distogram": {
                        "no_bins": 64,
                        "bin_min": 2.0,
                        "bin_max": 22.0,
                        "eps": eps,
                    },
                },
            },
        },
        # What a single dataset config should look like
        "dataset_config_template": {
            "name": "Placeholder name",
            "class": "Placeholder class",
            "mode": "Placeholder mode",
            "weight": 0.0,
            "config": {
                "n_templates": 4,
                "use_alignment_database": True,
                "loss_weight_mode": "default",
                "token_budget": 384,
                "crop_weights": {
                    "contiguous": 0.2,
                    "spatial": 0.4,
                    "spatial_interface": 0.4,
                },
                "loss": {
                    "min_resolution": 0.1,
                    "max_resolution": 4.0,
                    "confidence_loss_names": [
                        "plddt",
                        "pde",
                        "experimentally_resolved",
                        "pae",
                    ],
                    "loss_weights": {
                        "bond": 0.0,  # varies based on training and finetuning
                        "smooth_lddt": 4.0,  # varies based on finetuning stage
                        "mse": 4.0,
                        "distogram": 3e-2,
                        "experimentally_resolved": 0.0,
                        "plddt": 1e-4,
                        "pae": 0.0,
                        "pde": 1e-4,
                    },
                },
                "custom": {},
                "dataset_paths": {
                    "alignments_directory": PLACEHOLDER_PATH,
                    "target_structures_directory": PLACEHOLDER_PATH,
                    "alignment_db_directory": PLACEHOLDER_PATH,
                    "dataset_cache_file": PLACEHOLDER_PATH,
                    "reference_molecule_directory": PLACEHOLDER_PATH,
                    "template_cache_directory": PLACEHOLDER_PATH,
                    "template_structures_directory": PLACEHOLDER_PATH,
                    "ccd_file": PLACEHOLDER_PATH,
                },
            },
        },
        "extra_configs": {
            "loss_weight_modes": {
                "default": {
                    "bond": 0.0,  # varies based on training and finetuning
                    "smooth_lddt": 4.0,  # varies based on finetuning stage
                    "mse": 4.0,
                    "distogram": 3e-2,
                    "experimentally_resolved": 1e-4,
                    "plddt": 1e-4,
                    "pae": 0.0,
                    "pde": 1e-4,
                },
                # Custom losses will be applied as updates to the default loss
                "custom": {
                    "self_distillation": {
                        "experimentally_resolved": 0.0,
                        "pae": 0.0,
                        "plddt": 0.0,
                        "pde": 0.0,
                    },
                },
            },
        },
    }
)
