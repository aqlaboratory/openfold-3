import argparse
import logging
import os
import sys

import pytorch_lightning as pl
from ml_collections import ConfigDict

from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import DataModule
from openfold3.projects import registry
import torch
from tqdm import tqdm

if __name__ == "__main__":
    runner_args = ConfigDict(
        config_utils.load_yaml(
                "/global/cfs/cdirs/m4351/aqabel/openfold3/openfold3/openfold3/examples/runner.yml"
                )
                )
    project_entry = registry.get_project_entry(runner_args.project_type)
    project_config = registry.make_config_with_presets(project_entry, runner_args.presets)
    if runner_args.get("config_update"):
        project_config.update(runner_args.config_update)

    ckpt_path = None

    model_config = project_config.model
    lightning_module = project_entry.model_runner(
        model_config, _compile=runner_args.compile
    )

    dataset_config_builder = project_entry.dataset_config_builder
    data_module_config = registry.make_dataset_module_config(
        runner_args,
        dataset_config_builder,
        project_config,
    )
    lightning_data_module = DataModule(data_module_config)
    
    val_dataloader_iter = iter(lightning_data_module.val_dataloader())
    sample = next(val_dataloader_iter)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_module = lightning_module.to(device)
    lightning_module.eval()

    # device = 'cpu'
    for _ in tqdm(range(10)):
        lightning_module.tracker.increment()
        sample = next(val_dataloader_iter)
        dim = sample['residue_index'].shape[1]
        if dim > 600:
            print("Skipping, very large sample, shape: ", sample['residue_index'].shape)
            continue
        sample = lightning_module.transfer_batch_to_device(sample, device, 0)
        output = lightning_module.eval_step(sample, 0)
        del output
        torch.cuda.empty_cache()
    
    print(lightning_module.val_metrics.compute())