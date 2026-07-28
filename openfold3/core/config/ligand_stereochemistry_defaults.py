"""Default settings for inference-time ligand stereochemistry guidance."""

# Guidance is opt-in because it changes the reverse-diffusion trajectory.
DEFAULT_LIGAND_STEREOCHEMISTRY_GUIDANCE_ENABLED = False

# Begin guidance after 72.5% of the reverse-diffusion trajectory. This leaves
# early pose exploration unchanged while correcting local geometry before the
# final structure is written.
DEFAULT_LIGAND_STEREOCHEMISTRY_START_FRACTION = 0.725

# Number of analytic gradient updates applied to each guided denoised estimate.
DEFAULT_LIGAND_STEREOCHEMISTRY_NUM_GD_STEPS = 20

# Flat-bottom buffers and guidance weights for the physical potentials.
# Angular buffers are in radians.
BOND_BUFFER = 0.125
ANGLE_BUFFER = 0.125
CLASH_BUFFER = 0.10
CHIRAL_BUFFER = 0.52360
STEREO_BOND_BUFFER = 0.52360
PLANAR_BOND_BUFFER = 0.26180

POSEBUSTERS_WEIGHT = 0.01
CHIRAL_ATOM_WEIGHT = 0.1
STEREO_BOND_WEIGHT = 0.05
PLANAR_BOND_WEIGHT = 0.05

# Numerical floor used by the analytic distance and dihedral derivatives.
GEOMETRY_EPS = 1e-6

# Offset added to the mean RDKit van der Waals radius for each atom pair.
VDW_PAIR_CUTOFF_OFFSET = 0.35
