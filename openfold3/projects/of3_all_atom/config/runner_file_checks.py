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

from openfold3.core.data.framework.data_module import DataModuleConfig
from openfold3.projects.of3_all_atom.config.dataset_configs import TrainingDatasetSpec


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
