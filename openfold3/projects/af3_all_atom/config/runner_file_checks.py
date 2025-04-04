from openfold3.core.data.framework.data_module import DataModuleConfig
from openfold3.projects.af3_all_atom.config.dataset_configs import TrainingDatasetSpec


def _check_protein_monomer_sampled_in_order(dataset_config: TrainingDatasetSpec):
    """Check that monomer datasets are configured to be sampled in order"""
    if dataset_config.dataset_class == "ProteinMonomerDataset" and (
        not dataset_config.config.custom.sample_in_order
    ):
        raise ValueError(
            f"{dataset_config.name} is a monomer dataset, but is"
            "not configured to be sampled in order"
        )


def _check_data_module_config(data_module_config: DataModuleConfig):
    """Sanity checks for the data module config."""
    # Check dataset paths are valid for  key groups
    for dataset_cfg in data_module_config.datasets:
        # Check that deterministic sampling has been selected
        _check_protein_monomer_sampled_in_order(dataset_cfg)
