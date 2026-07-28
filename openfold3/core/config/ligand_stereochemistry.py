"""Configuration for inference-time ligand stereochemistry guidance."""

import math

from pydantic import BaseModel, Field
from pydantic import ConfigDict as PydanticConfigDict


class LigandStereochemistryGuidanceSettings(BaseModel):
    """Controls local ligand-geometry guidance during reverse diffusion."""

    model_config = PydanticConfigDict(extra="forbid", allow_inf_nan=False)

    start_fraction: float = Field(default=0.725, ge=0.0, le=1.0)
    num_gd_steps: int = Field(default=20, ge=1)

    bond_buffer: float = Field(default=0.125, ge=0.0, lt=1.0)
    angle_buffer: float = Field(default=0.125, ge=0.0, lt=1.0)
    clash_buffer: float = Field(default=0.10, ge=0.0, lt=1.0)
    chiral_buffer: float = Field(default=0.52360, ge=0.0, le=math.pi)
    stereo_bond_buffer: float = Field(default=0.52360, ge=0.0, le=math.pi)
    planar_bond_buffer: float = Field(default=0.26180, ge=0.0, le=math.pi)

    distance_weight: float = Field(default=0.01, ge=0.0)
    chiral_atom_weight: float = Field(default=0.1, ge=0.0)
    stereo_bond_weight: float = Field(default=0.05, ge=0.0)
    planar_bond_weight: float = Field(default=0.05, ge=0.0)

    vdw_pair_cutoff_offset: float = Field(default=0.35, ge=0.0)
