"""Default settings for dataset configurations for all atom project

The main sections of the dataset configuration are:
- MSA processing
- Templates
- Crops
- Loss
"""

from typing import Annotated

from pydantic import BaseModel, BeforeValidator

from openfold3.core.config.config_utils import _convert_molecule_type
from openfold3.core.data.resources.residues import MoleculeType


class MSASettings(BaseModel):
    """Settings for processing MSA features."""

    max_rows_paired: int = 8191
    max_rows: int = 16384
    subsample_with_bands: bool = False
    min_chains_paired_partial: int = 2
    pairing_mask_keys: list[str] = ["shared_by_two", "less_than_600"]
    moltypes: Annotated[list[MoleculeType], BeforeValidator(_convert_molecule_type)] = [
        MoleculeType.PROTEIN,
        MoleculeType.RNA,
    ]
    max_seq_counts: dict = {
        "uniref90_hits": 10000,
        "uniprot_hits": 50000,
        "bfd_uniclust_hits": 10000000,
        "bfd_uniref_hits": 10000000,
        "cfdb_uniref30": 10000000,
        "mgnify_hits": 5000,
        "rfam_hits": 10000,
        "rnacentral_hits": 10000,
        "nt_hits": 10000,
        "concat_cfdb_uniref100_filtered": 10000000,
    }
    msas_to_pair: list[str] = ["uniprot_hits", "uniprot"]
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
        "colabfold_main",
    ]
    paired_msa_order: list = ["colabfold_paired"]


colabfold_msa_settings = MSASettings(
    max_seq_counts={"colabfold_main": 16384, "colabfold_paired": 8192},
    moltypes=["protein", "rna"],
    max_rows_paired=8191,
    min_chains_paired_partial=2,
    aln_order=["colabfold_main"],
    paired_msa_order=["colabfold_paired"],
    msas_to_pair=[],
    pairing_mask_keys=["shared_by_two", "less_than_600"],
)


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
