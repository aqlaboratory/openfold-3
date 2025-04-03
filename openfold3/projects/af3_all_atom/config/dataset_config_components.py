"""Default settings for dataset configurations for all atom project

The main sections of the dataset configuration are:
- MSA processing 
- Templates 
- Crops
- Loss
"""
from pydantic import BaseModel


class MSAMaxSeqCounts(BaseModel):
    uniref90_hits: int = 10000
    uniprot_hits: int = 50000
    bfd_uniclust_hits: int = 10000000
    bfd_uniref_hits: int = 10000000
    cfdb_uniref30: int = 10000000
    mgnify_hits: int = 5000
    rfam_hits: int = 10000
    rnacentral_hits: int = 10000
    nt_hits: int = 10000
    concat_cfdb_uniref100_filtered: int = 10000000


class MSASettings(BaseModel):
    """Settings for processing MSA features."""
    max_rows_paired: int = 8191
    max_rows: int = 16384
    subsample_with_bands: bool = False
    min_chains_paired_partial: int = 2
    pairing_mask_keys: list[str] = ["shared_by_two", "less_than_600"]
    moltypes: list[str] = ["PROTEIN", "RNA"]
    max_seq_counts: MSAMaxSeqCounts = MSAMaxSeqCounts()
    aln_order: list = [
        "uniref90_hits",
        "bfd_uniclust_hits",
        "bfd_uniref_hits",
        "cfdb_uniref30",
        "mgnify_hits",
        "rfam_hits",
        "rnacentral_hits",
        "nt_hits",
        "concat_cfdb_uniref100_filtered",
    ]


class TemplateDistogramSettings(BaseModel):
    min_bin: float = 3.25
    max_bin: float = 50.75
    n_bins: int = 39


class TemplateSettings(BaseModel):
    """Settings for processing Template features."""
    n_templates: int = 4
    take_top_k: bool = False
    distogram: TemplateDistogramSettings = TemplateDistogramSettings()


class CropWeights(BaseModel):
    contiguous: float = 0.2
    spatial: float = 0.4
    spatial_interface: float = 0.4


class CropSettings(BaseModel):
    """Settings for crop featurization."""
    token_budget: int = 384
    crop_weights: CropWeights = CropWeights()


class LossWeights(BaseModel):
    bond: float = 0.0
    smooth_lddt: float = 4.0
    mse: float = 4.0
    distogram: float = 3e-2
    experimentally_resolved: float = 0.0
    plddt: float = 1e-4
    pae: float = 0.0
    pde: float = 1e-4


class LossConfig(BaseModel):
    """Settings for loss weights."""
    min_resolution: float = 0.1
    max_resolution: float = 4.0
    confidence_loss_names: list[str] = [
        "plddt",
        "pde",
        "experimentally_resolved",
        "pae",
    ]
    loss_weights: LossWeights = LossWeights()
