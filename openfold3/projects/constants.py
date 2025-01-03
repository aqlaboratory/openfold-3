"""
Constants specific to AF3 project. This currently only contains losses and
metrics for logging.
"""

LOSSES = [
    # Diffusion losses
    "mse_loss",
    "smooth_lddt_loss",
    "bond_loss",
    "diffusion_loss",
    # Confidence losses
    "plddt_loss",
    "pde_loss",
    "experimentally_resolved_loss",
    "confidence_loss",
    # Distogram losses
    "distogram_loss",
    "scaled_distogram_loss",
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
    # Superimposition metrics
    "superimpose_rmsd",
    "gdt_ts",
    "gdt_ha",
]

LOGGED_METRICS = METRICS + LOSSES
