from pathlib import Path

import torch

from openfold3.core.metrics.validation_all_atom import (
    get_superimpose_metrics,
    get_validation_metrics,
)
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.core.utils.tensor_utils import tensor_tree_map
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

        gt_coords = batch["ground_truth"]["atom_positions"].float()
        pred_coords = outputs["x_pred"].float()
        all_atom_mask = batch["ref_mask"]
        token_mask = batch["token_mask"]
        num_atoms_per_token = batch["num_atoms_per_token"]

        #getting rid of modified residues
        is_protein = batch["is_protein"] 
        is_rna = batch["is_rna"]
        is_dna = batch["is_dna"]
        not_modified_res = (1 - batch['is_atomized'])

        is_protein = is_protein * not_modified_res
        is_rna = is_rna * not_modified_res
        is_dna = is_dna * not_modified_res

        #broadcast token level features to atom level features
        is_protein_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                            num_atoms_per_token, 
                                                            is_protein)
        is_ligand_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                           num_atoms_per_token, 
                                                           batch['is_ligand'])
        is_rna_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                        num_atoms_per_token, 
                                                        is_rna)
        is_dna_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                        num_atoms_per_token, 
                                                        is_dna)
        asym_id_atomized = broadcast_token_feat_to_atoms(token_mask, 
                                                         num_atoms_per_token, 
                                                         batch['asym_id'])

        #get metrics
        protein_validation_metrics = get_validation_metrics(is_protein_atomized, 
                                                            asym_id_atomized,
                                                            pred_coords, 
                                                            gt_coords, 
                                                            all_atom_mask,
                                                            is_protein_atomized,
                                                            substrate = 'protein',
                                                            )
        metrics = metrics | protein_validation_metrics

        ligand_validation_metrics = get_validation_metrics(is_ligand_atomized, 
                                                           asym_id_atomized,
                                                           pred_coords, 
                                                           gt_coords, 
                                                           all_atom_mask,
                                                           is_protein_atomized,
                                                           substrate = 'ligand',
                                                           )
        metrics = metrics | ligand_validation_metrics 

        rna_validation_metrics = get_validation_metrics(is_rna_atomized, 
                                                        asym_id_atomized,
                                                        pred_coords, 
                                                        gt_coords, 
                                                        all_atom_mask,
                                                        is_protein_atomized,
                                                        substrate = 'rna',
                                                        )
        metrics = metrics | rna_validation_metrics

        dna_validation_metrics = get_validation_metrics(is_dna_atomized, 
                                                        asym_id_atomized,
                                                        pred_coords, 
                                                        gt_coords, 
                                                        all_atom_mask,
                                                        is_protein_atomized,
                                                        substrate = 'dna',
                                                        )
        metrics = metrics | dna_validation_metrics

        if superimposition_metrics:
            superimpose_metrics = get_superimpose_metrics(pred_coords,
                                                          gt_coords,
                                                          all_atom_mask,
                                                          )
            metrics = metrics | superimpose_metrics

        return metrics