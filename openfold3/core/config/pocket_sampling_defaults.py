"""Default settings for pocket-guided ligand proposal sampling."""

# Default ligand-to-pocket contact threshold in Angstroms. This is applied in
# the query schema so users can specify only ligand_chain_id and pocket_residues;
# the sampler still receives an explicit pocket_sampling_contact_distance tensor.
DEFAULT_POCKET_CONSTRAINT_MAX_DISTANCE = 4.0

# Presence of pocket_constraint enables proposal sampling by default. The
# environment override remains useful for ablations without changing input JSON.
DEFAULT_POCKET_SAMPLING_ENABLED = True

# Generate seeds from the best de-novo rollout parents. The parent count is
# capped by no_rollout_samples at runtime, so 16 is an upper bound for typical
# inference settings rather than a required number of samples.
DEFAULT_POCKET_SAMPLING_NUM_PARENTS = 16

# Number of random rigid ligand proposals to rank before diversity filtering.
# Proposal generation is cheap relative to diffusion, and 1024 gives enough
# coverage of rotations and pocket offsets without dominating runtime.
DEFAULT_POCKET_SAMPLING_CANDIDATES = 1024

# Start the refinement pass three quarters of the way through the diffusion
# schedule. This keeps the ligand in the proposed pocket basin while leaving
# enough noise for local protein-ligand relaxation.
DEFAULT_POCKET_SAMPLING_NOISE_FRAC = 0.75

# Small ligand-only coordinate jitter applied before refinement so repeated
# samples from the same seed do not follow identical local trajectories.
DEFAULT_POCKET_SAMPLING_LIGAND_JITTER = 0.25

# Pocket-center and pocket-surface proposal jitter in Angstroms. The broader
# center jitter explores placements around the residue set centroid; the smaller
# surface jitter explores local contacts around individual pocket atoms.
DEFAULT_POCKET_SAMPLING_CENTER_JITTER = 4.0
DEFAULT_POCKET_SAMPLING_SURFACE_JITTER = 1.5

# Clash screening uses van der Waals radii multiplied by (1 - buffer). A 0.225
# buffer allows imperfect generated poses while rejecting severe overlaps.
DEFAULT_POCKET_SAMPLING_VDW_BUFFER = 0.225

# Minimum heavy-atom RMSD between selected ligand proposals, in Angstroms.
DEFAULT_POCKET_SAMPLING_DIVERSITY_RMSD = 0.5

# Optional RDKit conformer ensemble size and deterministic embedding settings.
# Pruning is disabled so the OF3 proposal ranking, not RDKit, controls diversity.
DEFAULT_POCKET_SAMPLING_NUM_CONFORMERS = 32
DEFAULT_POCKET_SAMPLING_CONFORMER_RNG = 0
DEFAULT_POCKET_SAMPLING_CONFORMER_PRUNE_RMSD = 0.0
DEFAULT_POCKET_SAMPLING_CONFORMER_MAX_ITERS = 200
