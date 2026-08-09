"""Configuration for inference-time ligand stereochemistry guidance."""

import math

from pydantic import BaseModel, Field
from pydantic import ConfigDict as PydanticConfigDict


class LigandStereochemistryGuidanceSettings(BaseModel):
    """Controls local ligand-geometry guidance during reverse diffusion.

    Attributes:
        start_fraction (float):
            Fraction of reverse diffusion completed before guidance begins. Zero
            applies guidance throughout denoising and one applies it only to the
            final clean-coordinate estimate.
        num_gd_steps (int):
            Number of analytical coordinate updates applied at each guided step.
        bond_buffer (float):
            Relative flat-bottom tolerance around RDKit bonded-distance bounds.
        angle_buffer (float):
            Relative flat-bottom tolerance around RDKit 1-3 distance bounds.
        clash_buffer (float):
            Relative tolerance applied to nonbonded lower bounds before enforcing
            the van der Waals cutoff.
        chiral_buffer (float):
            Minimum signed tetrahedral improper-dihedral magnitude in radians.
        stereo_bond_buffer (float):
            Angular tolerance in radians around assigned E/Z configurations.
        planar_bond_buffer (float):
            Angular tolerance in radians around planar double-bond impropers.
        distance_weight (float):
            Coordinate-update weight for distance restraints.
        chiral_atom_weight (float):
            Coordinate-update weight for tetrahedral chirality restraints.
        stereo_bond_weight (float):
            Coordinate-update weight for E/Z restraints.
        planar_bond_weight (float):
            Coordinate-update weight for double-bond planarity restraints.
        vdw_pair_cutoff_offset (float):
            Offset in Angstroms added to the mean pairwise van der Waals radius.
    """

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
