# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    """Settings for processing MSA features.

    Attributes:
        max_rows_paired (int):
            Maximum number of rows for paired MSAs in heteromeric assemblies.
        max_rows (int):
            Maximum number of rows for MSA features including the query sequence +
            paired rows + unpaired rows.
        subsample_with_bands (bool):
            Whether to perform MMSeqs2-style subsampling at different sequence identity
            bands relative to the query sequence. Not currently supported.
        min_chains_paired_partial (int):
            Minimum number of chains for which to generate partially paired rows during
            online pairing. For example, if set to 3 and the query complex has 7 unique
            chains, then paired rows will be generated all 7 chains, any 6 of the 7
            chains ... down to any 3 of the 7 chains.
        pairing_mask_keys (list[str]):
            Masks to apply during online pairing to exclude certain sequences.
        moltypes (list[MoleculeType]):
            Molecule types to generate MSA features for. Only "protein" and "rna" are
            supported.
        max_seq_counts (dict):
            Maximum number of sequences to use from each MSA file specified by the
            corresponding key
        msas_to_pair (list[str]):
            Designated MSA files to use for online pairing. Requires species information
            to be present in the MSA files in the format outlined in the Precomputed MSA
            How-To Guide.
        aln_order (list):
            The order in which to vertically concatenate the MSA files for each chain.
        paired_msa_order (list):
            The order in which to vertically concatenate pre-paired MSA files for each
            chain, if provided.
    """

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
        "colabfold_main": 16384,
        "colabfold_paired": 8192,
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
        "dummy",  # aln containing only query; used for MSA-free inference
    ]
    paired_msa_order: list = ["colabfold_paired"]


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
    experimentally_resolved: float = 1e-4
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
