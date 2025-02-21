from ml_collections import ConfigDict

from openfold3.core.data.framework.data_module import DataModuleConfig


def _check_file_path_group(dataset_path_config, dataset_path_names):
    paths = [dataset_path_config[k] for k in dataset_path_names]
    if not (any(paths)):
        raise ValueError(f"No paths set amongst {dataset_path_names}")

    for path in paths:
        if path and not path.exists():
            raise ValueError(f"{path} does not exist")


def _check_protein_monomer_sampled_in_order(dataset_config: ConfigDict):
    """Check that monomer datasets are configured to be sampled in order"""
    if dataset_config["class"] == "ProteinMonomerDataset" and (
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
        dataset_paths = dataset_cfg.config.dataset_paths

        alignment_path_names = [
            "alignments_directory",
            "alignment_db_directory",
            "alignment_array_directory",
        ]
        _check_file_path_group(dataset_paths, alignment_path_names)

        template_path_names = [
            "template_structures_directory",
            "template_structure_array_directory",
        ]
        _check_file_path_group(dataset_paths, template_path_names)

        structure_path_names = ["target_structures_directory"]
        _check_file_path_group(dataset_paths, structure_path_names)

        # Check that deterministic sampling has been selected
        _check_protein_monomer_sampled_in_order(dataset_cfg)
