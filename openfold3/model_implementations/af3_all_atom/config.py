import copy
import importlib

import ml_collections as mlc


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def enforce_config_constraints(config):
    def string_to_setting(s):
        path = s.split(".")
        setting = config
        for p in path:
            setting = setting.get(p)

        return setting

    mutually_exclusive_bools = [
        ("globals.use_lma", "globals.use_flash", "globals.use_deepspeed_evo_attention"),
    ]

    for options in mutually_exclusive_bools:
        option_settings = [string_to_setting(o) for o in options]
        if sum(option_settings) > 1:
            raise ValueError(f"Only one of {', '.join(options)} may be set at a time")

    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    if config.globals.use_flash and not fa_is_installed:
        raise ValueError("use_flash requires that FlashAttention is installed")

    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    ds4s_is_installed = (
        deepspeed_is_installed
        and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
    )
    if config.globals.use_deepspeed_evo_attention and not ds4s_is_installed:
        raise ValueError(
            "use_deepspeed_evo_attention requires that DeepSpeed be installed "
            "and that the deepspeed.ops.deepspeed4science package exists"
        )


def model_config(
    name,
    train=False,
    low_prec=False,
    long_sequence_inference=False,
    use_deepspeed_evoformer_attention=False,
):
    c = copy.deepcopy(config)

    # TODO: Named model configs unless this is moved somewhere else

    if long_sequence_inference:
        assert not train
        c.globals.offload_inference = True
        # Default to DeepSpeed memory-efficient attention kernel unless use_lma
        # is explicitly set
        c.globals.use_deepspeed_evo_attention = bool(not c.globals.use_lma)
        c.globals.use_flash = False
        c.model.template.offload_inference = True
        c.model.template.template_pair_stack.tune_chunk_size = False
        c.model.msa.msa_module.tune_chunk_size = False
        c.model.pairformer.tune_chunk_size = False

    if use_deepspeed_evoformer_attention:
        c.globals.use_deepspeed_evo_attention = True

    if train:
        c.globals.blocks_per_ckpt = 1
        c.globals.chunk_size = None
        c.globals.use_lma = False
        c.globals.offload_inference = False

    if low_prec:
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)

    enforce_config_constraints(c)

    return c


NUM_TOKENS = "num tokens placeholder"
NUM_ATOMS = "num atoms placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_TEMPLATES = "num templates placeholder"


# Hidden dimensions
c_s = mlc.FieldReference(384, field_type=int)
c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(64, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_atom_ref = mlc.FieldReference(390, field_type=int)
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

# templates_enabled = mlc.FieldReference(True, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
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
                    "atom_to_token_index": [NUM_ATOMS, NUM_TOKENS],
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
                "max_relative_idx": max_relative_idx,
                "max_relative_chain": max_relative_chain,
                "inf": 1e10,  # global parameter?
            },
            "template": {
                "c_t": c_t,
                "c_z": c_z,
                "template_pair_embedder": {
                    "c_in": 108,
                    "c_z": c_z,
                    "c_out": c_t,
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
                    "tune_chunk_size": tune_chunk_size,
                    "inf": 1e9,
                },
            },
            "msa": {
                "msa_module_embedder": {
                    "c_m_feats": 34,
                    "c_m": c_m,
                    "c_s_input": c_s_input,
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
                "clear_cache_between_blocks": False,
                "tune_chunk_size": tune_chunk_size,
            },
            "diffusion_module": {
                "diffusion_module": {
                    "c_s": c_s,
                    "c_token": c_token_diffusion,
                    "sigma_data": sigma_data,
                },
                "diffusion_conditioning": {
                    "c_s_input": c_s_input,
                    "c_s": c_s,
                    "c_z": c_z,
                    "sigma_data": sigma_data,
                    "c_fourier_emb": 256,
                    "max_relative_idx": max_relative_idx,
                    "max_relative_chain": max_relative_chain,
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
                    "inf": 1e9,  # global parameter?
                },
                "diffusion_transformer": {
                    "c_a": c_token_diffusion,
                    "c_s": c_s,
                    "c_z": c_z,
                    "c_hidden": 48,  # c_token / no_heads
                    "no_heads": 16,
                    "no_blocks": 24,
                    "n_transition": 2,
                    "inf": 1e9,  # global parameter?
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
                    "inf": 1e9,  # global parameter?
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
        },
        "loss": {
            "diffusion": {
                "sigma_data": sigma_data,
                "alpha_bond": 0.0,  # varies based on training and finetuning
                "alpha_dna": 5.0,
                "alpha_rna": 5.0,
                "alpha_ligand": 10.0,
            }
        },
    }
)

train_config_update = mlc.ConfigDict({"loss": {"diffusion": {"alpha_bond": 0.0}}})

finetune1_config_update = mlc.ConfigDict({"loss": {"diffusion": {"alpha_bond": 1.0}}})

finetune2_config_update = mlc.ConfigDict({"loss": {"diffusion": {"alpha_bond": 1.0}}})

eval_config_update = mlc.ConfigDict({"globals": {"no_rollout_steps": 200}})
