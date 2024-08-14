from pathlib import Path

import torch

from openfold3.core.metrics.validaiton_all_atom import (
    get_pair_dist,
    get_superimpose_metrics,
    get_validation_metrics,
    interface_lddt,
    lddt,
)
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.model_implementations.af3_all_atom.config.base_config import config
from openfold3.model_implementations.af3_all_atom.model import AlphaFold3
from openfold3.model_implementations.registry import register_model

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


@register_model("af3_all_atom", config, REFERENCE_CONFIG_PATH)
class AlphaFold3AllAtom(ModelRunner):
    def __init__(self, model_config):
        super().__init__(AlphaFold3, model_config)

    def _compute_validation_metrics(
        self, batch, outputs, superimposition_metrics=False
    ):
        metrics = {}

        gt_coords = batch["Ground_Truth"]["ref_pos"] #confirm the name
        pred_coords = outputs["x_pred"]
        all_atom_mask = batch["ref_mask"]
        token_mask = batch["token_mask"]
        num_atoms_per_token = batch["Ground_Truth"]["num_atoms_per_token"]

        #get rid of modified residue tokens 
        is_protein = batch["Groud_Truth"]["is_protein"] 
        is_rna = batch["Groud_Truth"]["is_rna"] 
        is_dna = batch["Groud_Truth"]["is_dna"] 
        not_modified_res = (1 - batch["Groud_Truth"]['is_atomized'])

        is_protein = is_protein * not_modified_res
        is_rna = is_rna * not_modified_res
        is_dna = is_dna * not_modified_res

        is_protein_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                            num_atoms_per_token, 
                                                            is_protein)
        is_ligand_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                           num_atoms_per_token, 
                                                           batch["Groud_Truth"]['is_ligand'])
        is_rna_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                        num_atoms_per_token, 
                                                        is_rna)
        is_dna_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                        num_atoms_per_token, 
                                                        is_dna)
        asym_id_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                         num_atoms_per_token, 
                                                         batch["Groud_Truth"]['asym_id'])
        protein_idx = torch.nonzero(is_protein_atomized).squeeze(-1)

        #get metrics
        protein_validation_metrics = get_validation_metrics(is_protein_atomized, 
                                                            asym_id_atomized,
                                                            pred_coords, 
                                                            gt_coords, 
                                                            all_atom_mask,
                                                            protein_idx,
                                                            ligand_type = 'protein',
                                                            is_nucleic_acid = False,
                                                            )
        metrics = metrics | protein_validation_metrics

        ligand_validation_metrics = get_validation_metrics(is_ligand_atomized, 
                                                           asym_id_atomized,
                                                           pred_coords, 
                                                           gt_coords, 
                                                           all_atom_mask,
                                                           protein_idx,
                                                           ligand_type = 'ligand',
                                                           is_nucleic_acid = False,
                                                           )
        metrics = metrics | ligand_validation_metrics 

        rna_validation_metrics = get_validation_metrics(is_rna_atomized, 
                                                        asym_id_atomized,
                                                        pred_coords, 
                                                        gt_coords, 
                                                        all_atom_mask,
                                                        protein_idx,
                                                        ligand_type = 'rna',
                                                        is_nucleic_acid = True,
                                                        )
        metrics = metrics | rna_validation_metrics

        dna_validation_metrics = get_validation_metrics(is_dna_atomized, 
                                                        asym_id_atomized,
                                                        pred_coords, 
                                                        gt_coords, 
                                                        all_atom_mask,
                                                        protein_idx,
                                                        ligand_type = 'dna',
                                                        is_nucleic_acid = True,
                                                        )
        metrics = metrics | dna_validation_metrics

        #extra LDDTs for model selections:
        #interLDDT: ligand_dna. Should I bother including these metrics??? also, 
        if torch.any(is_ligand_atomized) and torch.any(is_dna_atomized):
            metrics.update({'lddt_inter_ligand_dna': interface_lddt(
                pred_coords[is_ligand_atomized],
                pred_coords[is_dna_atomized],
                gt_coords[is_ligand_atomized],
                gt_coords[is_dna_atomized],
                all_atom_mask[is_ligand_atomized], 
                all_atom_mask[is_dna_atomized], 
                cutoff= 30.,
                )})
        #interLDDT: ligand_rna
        if torch.any(is_ligand_atomized) and torch.any(is_rna_atomized):
            metrics.update({'lddt_inter_ligand_rna': interface_lddt(
                pred_coords[is_ligand_atomized],
                pred_coords[is_rna_atomized],
                gt_coords[is_ligand_atomized],
                gt_coords[is_rna_atomized],
                all_atom_mask[is_ligand_atomized], 
                all_atom_mask[is_rna_atomized], 
                cutoff= 30.,
                )})
            
        #intraLDDT: modified_residues
        is_modified_res = batch["Groud_Truth"]['is_atomized']
        is_modified_res = is_modified_res * (1 - batch["Groud_Truth"]['is_ligand'])
        is_modified_res_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                                 num_atoms_per_token, 
                                                                 is_modified_res)
        pred_modified_res_pair = get_pair_dist(pred_coords[is_modified_res_atomized],
                                               pred_coords[is_modified_res_atomized])
        gt_modified_res_pair = get_pair_dist(gt_coords[is_modified_res_atomized],
                                             gt_coords[is_modified_res_atomized])
        intra_mod_res_lddt, _ = lddt(pred_modified_res_pair,
                                     gt_modified_res_pair,
                                     all_atom_mask[is_modified_res_atomized], 
                                     asym_id_atomized[is_modified_res_atomized])
        metrics.update({'lddt_intra_modified_residues': intra_mod_res_lddt})

        if superimposition_metrics:
            superimpose_metrics = get_superimpose_metrics(pred_coords,
                                                          gt_coords,
                                                          all_atom_mask,
                                                          )
            metrics = metrics | superimpose_metrics

        return metrics

