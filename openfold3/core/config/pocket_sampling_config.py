# Copyright 2026 AlQuraishi Laboratory
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

"""Settings for pocket-guided ligand proposal sampling."""

import math

from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict as PydanticConfigDict

# Default ligand-to-pocket contact threshold in Angstroms. This is applied in
# the query schema so users can specify only ligand_chain_id and pocket_residues;
# the sampler still receives an explicit pocket_sampling_contact_distance tensor.
DEFAULT_POCKET_CONSTRAINT_MAX_DISTANCE = 4.0


class PocketSamplingSettings(BaseModel):
    """Settings for pocket-guided ligand proposal sampling and refinement.

    Consumed by `create_pocket_sampling_features` in
    `openfold3.core.data.pipelines.featurization.pocket_constraints`. Reachable
    from an inference experiment config via
    `dataset_config_kwargs.pocket_sampling`.

    Attributes:
        enabled (bool):
            Whether pocket-guided proposal sampling runs when a query provides
            a pocket_constraint. Presence of pocket_constraint enables proposal
            sampling by default; this remains useful for ablations without
            changing input JSON.
        num_parents (int):
            Number of de-novo rollout parents to seed refinement from. The
            parent count is capped by no_rollout_samples at runtime, so this is
            an upper bound for typical inference settings rather than a
            required number of samples.
        candidates (int):
            Number of random rigid ligand proposals to rank before diversity
            filtering. Proposal generation is cheap relative to diffusion, so a
            large value gives coverage of rotations and pocket offsets without
            dominating runtime.
        noise_frac (float):
            Fraction of the diffusion schedule to complete before starting the
            refinement pass. Starting late keeps the ligand in the proposed
            pocket basin while leaving enough noise for local protein-ligand
            relaxation.
        ligand_jitter (float):
            Small ligand-only coordinate jitter (Angstroms) applied before
            refinement so repeated samples from the same seed do not follow
            identical local trajectories.
        center_jitter (float):
            Pocket-center proposal jitter (Angstroms) exploring placements
            around the residue set centroid.
        surface_jitter (float):
            Pocket-surface proposal jitter (Angstroms) exploring local contacts
            around individual pocket atoms.
        vdw_buffer (float):
            Clash screening uses van der Waals radii multiplied by
            (1 - vdw_buffer). Allows imperfect generated poses while rejecting
            severe overlaps.
        diversity_rmsd (float):
            Minimum heavy-atom RMSD (Angstroms) between selected ligand
            proposals.
        rdkit_num_conformers (int):
            Number of RDKit conformers to generate for the ligand ensemble. Set
            to 0 to disable RDKit conformer generation.
        rdkit_conformer_rng (int):
            Random seed for deterministic RDKit conformer embedding.
        rdkit_conformer_prune_rmsd (float):
            RDKit embedding RMSD pruning threshold. Disabled (0.0) by default so
            the OF3 proposal ranking, not RDKit, controls diversity.
        rdkit_conformer_max_iters (int):
            Maximum number of force-field optimization iterations per RDKit
            conformer.
    """

    model_config = PydanticConfigDict(extra="forbid")

    enabled: bool = True
    num_parents: int = Field(default=16, ge=1)
    candidates: int = Field(default=1024, ge=1)
    noise_frac: float = Field(default=0.75, ge=0.0, le=1.0)
    ligand_jitter: float = Field(default=0.25, ge=0.0)
    center_jitter: float = Field(default=4.0, ge=0.0)
    surface_jitter: float = Field(default=1.5, ge=0.0)
    vdw_buffer: float = Field(default=0.225, ge=0.0)
    diversity_rmsd: float = Field(default=0.5, ge=0.0)
    rdkit_num_conformers: int = Field(default=32, ge=0)
    rdkit_conformer_rng: int = 0
    rdkit_conformer_prune_rmsd: float = Field(default=0.0, ge=0.0)
    rdkit_conformer_max_iters: int = Field(default=200, ge=1)

    @field_validator(
        "noise_frac",
        "ligand_jitter",
        "center_jitter",
        "surface_jitter",
        "vdw_buffer",
        "diversity_rmsd",
        "rdkit_conformer_prune_rmsd",
    )
    @classmethod
    def _validate_finite(cls, value: float, info) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be a finite float; got {value!r}")
        return value
