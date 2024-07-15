import ml_collections as mlc

# All the contents from the original config.py relating to data processing
# are moved here for now. See where these fit in with the new data pipeline.
data_config_settings = mlc.ConfigDict(
    {
            "predict": {
                "fixed_size": True,
                "subsample_templates": False,  # We want top templates.
                "block_delete_msa": False,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 512,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "crop": False,
                "crop_size": None,
                "spatial_crop_prob": None,
                "interface_threshold": None,
                "supervised": False,
                "uniform_recycling": False,
            },
            "eval": {
                "fixed_size": True,
                "subsample_templates": False,  # We want top templates.
                "block_delete_msa": False,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "crop": False,
                "crop_size": None,
                "spatial_crop_prob": None,
                "interface_threshold": None,
                "supervised": True,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "subsample_templates": True,
                "block_delete_msa": True,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "shuffle_top_k_prefiltered": 20,
                "crop": True,
                "crop_size": 256,
                "spatial_crop_prob": 0.0,
                "interface_threshold": None,
                "supervised": True,
                "clamp_prob": 0.9,
                "max_distillation_msa_clusters": 1000,
                "uniform_recycling": True,
                "distillation_prob": 0.75,
            },
            "data_module": {
                "use_small_bfd": False,
                "data_loaders": {
                    "batch_size": 1,
                    "num_workers": 16,
                    "pin_memory": True,
                },
            },
    }
)
