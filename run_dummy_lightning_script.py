import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset

from openfold3.model_implementations import registry
from tests.data_utils import random_af3_features


class DummyAF3Dataset(IterableDataset):
    """A dummy AF3 dataset which creates a dataloader of dummy features"""

    def __init__(self):
        super(DummyAF3Dataset).__init__()
        self.config = registry.MODEL_REGISTRY["af3_all_atom"].base_config
        self.n_token = 16
        self.n_msa = 10
        self.n_templ = 3
    
    def create_random_features(self):
        yield random_af3_features(
            batch_size=1,
            n_token=self.n_token,
            n_msa=self.n_msa,
            n_templ=self.n_templ,
        )
    
    def __iter__(self):
        return self.create_random_features() 


class DummyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.mydataset = DummyAF3Dataset()

    def train_dataloader(self):
        return DataLoader(self.mydataset)


if __name__ == "__main__":
    lightning_data_module = DummyDataModule() 

    logging.info("Loading model")
    config = registry.make_config_with_preset("af3_all_atom")
    config.model.pairformer.no_blocks = 4
    config.model.diffusion_module.diffusion_transformer.no_blocks = 4
    lightning_module = registry.get_lightning_module(config)

    trainer = pl.Trainer(num_nodes=1, fast_dev_run=1)
    trainer.fit(lightning_module, lightning_data_module)