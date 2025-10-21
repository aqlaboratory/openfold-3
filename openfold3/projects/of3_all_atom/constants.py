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

"""
Constants specific to AF3 project. This currently only contains losses and
metrics for logging.
"""

###################################
# Losses
###################################

CONFIDENCE_LOSSES = [
    "plddt_loss",
    "pde_loss",
    "experimentally_resolved_loss",
    "pae_loss",
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

###################################
# Metrics
###################################

PROTEIN_METRICS = [
    "lddt_intra_protein",
    "lddt_inter_protein_protein",
    "drmsd_intra_protein",
    "clash_intra_protein",
    "clash_inter_protein_protein",
]

LIGAND_METRICS = [
    "lddt_intra_ligand",
    "lddt_inter_ligand_ligand",
    "lddt_intra_ligand_uha",
    "lddt_inter_ligand_ligand_uha",
    "lddt_inter_protein_ligand",
    "drmsd_intra_ligand",
    "clash_intra_ligand",
    "clash_inter_ligand_ligand",
    "clash_inter_protein_ligand",
]

DNA_METRICS = [
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
]

RNA_METRICS = [
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

METRICS = [
    *PROTEIN_METRICS,
    *LIGAND_METRICS,
    *DNA_METRICS,
    *RNA_METRICS,
]

###################################
# Training-specific Extra Metrics
###################################

# Mirrors ligand metrics but for diffusion training output
TRAIN_EXTRA_METRICS = [f"{k}_diffusion" for k in LIGAND_METRICS]

TRAIN_LOGGED_METRICS = [*METRICS, *TRAIN_EXTRA_METRICS]

###################################
# Model Selection
# pLDDT/LDDT Correlation Metrics
# Superimposition Metrics
###################################

SUPERIMPOSE_METRICS = [
    "rmsd",
    "gdt_ts",
    "gdt_ha",
]

VAL_EXTRA_METRICS = [
    *SUPERIMPOSE_METRICS,
    # RASA for model selection
    "rasa",
    # LDDT metrics for model selection
    "lddt_inter_ligand_dna",
    "lddt_inter_ligand_rna",
    "lddt_intra_modified_residues",
    # Complex metrics
    "lddt_intra_complex",
    # pLDDT metrics
    "plddt_protein",
    "plddt_ligand",
    "plddt_dna",
    "plddt_rna",
    "plddt_complex",
]

VAL_LOGGED_METRICS = [
    *METRICS,
    *VAL_EXTRA_METRICS,
]

CORRELATION_METRICS = [
    "pearson_correlation_lddt_plddt_protein",
    "pearson_correlation_lddt_plddt_ligand",
    "pearson_correlation_lddt_plddt_dna",
    "pearson_correlation_lddt_plddt_rna",
    "pearson_correlation_lddt_plddt_complex",
]

METRICS_MAXIMIZE = [
    "gdt",
    "lddt",
    "plddt",
    "rasa",
]

METRICS_MINIMIZE = [
    "clash",
    "drmsd",
    "rmsd",
]
