"""
Constants specific to AF3 project. This currently only contains losses and
metrics for logging.
"""

CONFIDENCE_LOSSES = [
    "plddt_loss",
    "pde_loss",
    "experimentally_resolved_loss",
    "confidence_loss",
]

DIFFUSION_LOSSES = [
    "mse_loss",
    "smooth_lddt_loss",
    "bond_loss",
    "diffusion_loss",
]

DISTOGRAM_LOSSES = [
    "distogram_loss",
    "scaled_distogram_loss",
]


TRAIN_LOSSES = [
    *CONFIDENCE_LOSSES,
    *DIFFUSION_LOSSES,
    *DISTOGRAM_LOSSES,
    "loss",
]

VAL_LOSSES = [
    *CONFIDENCE_LOSSES,
    *DISTOGRAM_LOSSES,
    "loss",
]

METRICS = [
    # Protein metrics
    "lddt_intra_protein",
    "lddt_inter_protein_protein",
    "drmsd_intra_protein",
    "clash_intra_protein",
    "clash_inter_protein_protein",
    # Ligand metrics
    "lddt_intra_ligand",
    "lddt_inter_ligand_ligand",
    "lddt_intra_ligand_uha",
    "lddt_inter_ligand_ligand_uha",
    "lddt_inter_protein_ligand",
    "drmsd_intra_ligand",
    "clash_intra_ligand",
    "clash_inter_ligand_ligand",
    "clash_inter_protein_ligand",
    # DNA metrics
    "lddt_intra_dna",
    "lddt_inter_dna_dna",
    "drmsd_intra_dna",
    "lddt_intra_dna_15",
    "lddt_inter_dna_dna_15",
    "lddt_inter_protein_dna",
    "lddt_inter_protein_dna_15",
    "clash_intra_dna",
    "clash_inter_dna_dna",
    "clash_inter_protein_dna",
    # RNA metrics
    "lddt_intra_rna",
    "lddt_inter_rna_rna",
    "drmsd_intra_rna",
    "lddt_intra_rna_15",
    "lddt_inter_rna_rna_15",
    "lddt_inter_protein_rna",
    "lddt_inter_protein_rna_15",
    "clash_intra_rna",
    "clash_inter_rna_rna",
    "clash_inter_protein_rna",
]

SUPERIMPOSE_METRICS = [
    "superimpose_rmsd",
    "gdt_ts",
    "gdt_ha",
]

VAL_EXTRA_LDDT_METRICS = [
    "lddt_inter_ligand_dna",
    "lddt_inter_ligand_rna",
    "lddt_intra_modified_residues",
]

VAL_EXTRA_LDDT_CORR_METRICS = [
    # pLDDT metrics
    "plddt_protein",
    "plddt_ligand",
    "plddt_dna",
    "plddt_rna",
    # Complex metrics
    "lddt_complex",
    "plddt_complex",
]

MODEL_SELECTION = [
    *VAL_EXTRA_LDDT_METRICS,
    "model_selection",
]

TRAIN_LOGGED_METRICS = [
    *TRAIN_LOSSES,
    *METRICS,
]

VAL_LOGGED_METRICS = [
    *VAL_LOSSES,
    *METRICS,
    *SUPERIMPOSE_METRICS,
    *MODEL_SELECTION,
    *VAL_EXTRA_LDDT_CORR_METRICS,
]
