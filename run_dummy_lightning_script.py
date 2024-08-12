import logging

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader, IterableDataset

from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.model_implementations import registry
from tests.data_utils import random_af3_features


class DummyAF3Dataset(IterableDataset):
    """A dummy AF3 dataset which creates a dataloader of dummy features"""

    def __init__(self):
        super(DummyAF3Dataset).__init__()
        self.config = registry.make_config_with_preset("af3_all_atom")

        self.n_token = 384

        # For full MSA size
        # self.n_msa = 16384
        self.n_msa = 8000

        self.n_templ = 4

    def create_random_features(self):
        feats = random_af3_features(
            batch_size=1,
            n_token=self.n_token,
            n_msa=self.n_msa,
            n_templ=self.n_templ,
        )

        # Batch dim is added by collator, quick fix to remove it
        # This will get replaced with actual dataloader anyway
        feats = tensor_tree_map(lambda x: x.squeeze(0), feats)
        yield feats

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
    config.globals.use_deepspeed_evo_attention = True
    config.model.input_embedder.atom_attn_enc.use_block_sparse_attn = True
    config.model.diffusion_module.atom_attn_enc.use_block_sparse_attn = True
    config.model.diffusion_module.atom_attn_dec.use_block_sparse_attn = True
    config.globals.blocks_per_ckpt = 1
    config.globals.chunk_size = None

    lightning_module = registry.get_lightning_module(config)

    strategy = DeepSpeedStrategy(
        config="deepspeed_config.json", cluster_environment=MPIEnvironment()
    )

    trainer = pl.Trainer(
        num_nodes=1,
        devices=4,
        fast_dev_run=1,
        precision="bf16-mixed",
        strategy=strategy,
    )

    trainer.fit(lightning_module, lightning_data_module)
