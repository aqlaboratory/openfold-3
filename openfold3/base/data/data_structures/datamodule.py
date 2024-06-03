# TODO add license

from functools import partial
import json
from typing import Optional

import ml_collections as mlc
import pytorch_lightning as pl
import torch


# TODO yet to refactor
# class OpenFoldDataModule(pl.LightningDataModule):
#     def __init__(self,
#                  config: mlc.ConfigDict,
#                  template_mmcif_dir: str,
#                  max_template_date: str,
#                  train_data_dir: Optional[str] = None,
#                  train_alignment_dir: Optional[str] = None,
#                  train_chain_data_cache_path: Optional[str] = None,
#                  distillation_data_dir: Optional[str] = None,
#                  distillation_alignment_dir: Optional[str] = None,
#                  distillation_chain_data_cache_path: Optional[str] = None,
#                  val_data_dir: Optional[str] = None,
#                  val_alignment_dir: Optional[str] = None,
#                  predict_data_dir: Optional[str] = None,
#                  predict_alignment_dir: Optional[str] = None,
#                  kalign_binary_path: str = '/usr/bin/kalign',
#                  train_filter_path: Optional[str] = None,
#                  distillation_filter_path: Optional[str] = None,
#                  obsolete_pdbs_file_path: Optional[str] = None,
#                  template_release_dates_cache_path: Optional[str] = None,
#                  batch_seed: Optional[int] = None,
#                  train_epoch_len: int = 50000,
#                  _distillation_structure_index_path: Optional[str] = None,
#                  alignment_index_path: Optional[str] = None,
#                  distillation_alignment_index_path: Optional[str] = None,
#                  **kwargs
#                  ):
#         super(OpenFoldDataModule, self).__init__()

#         self.config = config
#         self.template_mmcif_dir = template_mmcif_dir
#         self.max_template_date = max_template_date
#         self.train_data_dir = train_data_dir
#         self.train_alignment_dir = train_alignment_dir
#         self.train_chain_data_cache_path = train_chain_data_cache_path
#         self.distillation_data_dir = distillation_data_dir
#         self.distillation_alignment_dir = distillation_alignment_dir
#         self.distillation_chain_data_cache_path = (
#             distillation_chain_data_cache_path
#         )
#         self.val_data_dir = val_data_dir
#         self.val_alignment_dir = val_alignment_dir
#         self.predict_data_dir = predict_data_dir
#         self.predict_alignment_dir = predict_alignment_dir
#         self.kalign_binary_path = kalign_binary_path
#         self.train_filter_path = train_filter_path
#         self.distillation_filter_path = distillation_filter_path
#         self.template_release_dates_cache_path = (
#             template_release_dates_cache_path
#         )
#         self.obsolete_pdbs_file_path = obsolete_pdbs_file_path
#         self.batch_seed = batch_seed
#         self.train_epoch_len = train_epoch_len

#         if self.train_data_dir is None and self.predict_data_dir is None:
#             raise ValueError(
#                 'At least one of train_data_dir or predict_data_dir must be '
#                 'specified'
#             )

#         self.training_mode = self.train_data_dir is not None

#         if self.training_mode and train_alignment_dir is None:
#             raise ValueError(
#                 'In training mode, train_alignment_dir must be specified'
#             )
#         elif not self.training_mode and predict_alignment_dir is None:
#             raise ValueError(
#                 'In inference mode, predict_alignment_dir must be specified'
#             )
#         elif val_data_dir is not None and val_alignment_dir is None:
#             raise ValueError(
#                 'If val_data_dir is specified, val_alignment_dir must '
#                 'be specified as well'
#             )

#         # An ad-hoc measure for our particular filesystem restrictions
#         self._distillation_structure_index = None
#         if _distillation_structure_index_path is not None:
#             with open(_distillation_structure_index_path, "r") as fp:
#                 self._distillation_structure_index = json.load(fp)

#         self.alignment_index = None
#         if alignment_index_path is not None:
#             with open(alignment_index_path, "r") as fp:
#                 self.alignment_index = json.load(fp)

#         self.distillation_alignment_index = None
#         if distillation_alignment_index_path is not None:
#             with open(distillation_alignment_index_path, "r") as fp:
#                 self.distillation_alignment_index = json.load(fp)

#     def setup(self, stage=None):
#         # Most of the arguments are the same for the three datasets 
#         dataset_gen = partial(OpenFoldSingleDataset,
#                               template_mmcif_dir=self.template_mmcif_dir,
#                               max_template_date=self.max_template_date,
#                               config=self.config,
#                               kalign_binary_path=self.kalign_binary_path,
#                               template_release_dates_cache_path=self.template_release_dates_cache_path,
#                               obsolete_pdbs_file_path=self.obsolete_pdbs_file_path)

#         if self.training_mode:
#             train_dataset = dataset_gen(
#                 data_dir=self.train_data_dir,
#                 chain_data_cache_path=self.train_chain_data_cache_path,
#                 alignment_dir=self.train_alignment_dir,
#                 filter_path=self.train_filter_path,
#                 max_template_hits=self.config.train.max_template_hits,
#                 shuffle_top_k_prefiltered=self.config.train.shuffle_top_k_prefiltered,
#                 treat_pdb_as_distillation=False,
#                 mode="train",
#                 alignment_index=self.alignment_index,
#             )

#             distillation_dataset = None
#             if self.distillation_data_dir is not None:
#                 distillation_dataset = dataset_gen(
#                     data_dir=self.distillation_data_dir,
#                     chain_data_cache_path=self.distillation_chain_data_cache_path,
#                     alignment_dir=self.distillation_alignment_dir,
#                     filter_path=self.distillation_filter_path,
#                     max_template_hits=self.config.train.max_template_hits,
#                     treat_pdb_as_distillation=True,
#                     mode="train",
#                     alignment_index=self.distillation_alignment_index,
#                     _structure_index=self._distillation_structure_index,
#                 )

#                 d_prob = self.config.train.distillation_prob

#             if distillation_dataset is not None:
#                 datasets = [train_dataset, distillation_dataset]
#                 d_prob = self.config.train.distillation_prob
#                 probabilities = [1. - d_prob, d_prob]
#             else:
#                 datasets = [train_dataset]
#                 probabilities = [1.]

#             generator = None
#             if self.batch_seed is not None:
#                 generator = torch.Generator()
#                 generator = generator.manual_seed(self.batch_seed + 1)

#             self.train_dataset = OpenFoldDataset(
#                 datasets=datasets,
#                 probabilities=probabilities,
#                 epoch_len=self.train_epoch_len,
#                 generator=generator,
#                 _roll_at_init=False,
#             )

#             if self.val_data_dir is not None:
#                 self.eval_dataset = dataset_gen(
#                     data_dir=self.val_data_dir,
#                     alignment_dir=self.val_alignment_dir,
#                     filter_path=None,
#                     max_template_hits=self.config.eval.max_template_hits,
#                     mode="eval",
#                 )
#             else:
#                 self.eval_dataset = None
#         else:
#             self.predict_dataset = dataset_gen(
#                 data_dir=self.predict_data_dir,
#                 alignment_dir=self.predict_alignment_dir,
#                 filter_path=None,
#                 max_template_hits=self.config.predict.max_template_hits,
#                 mode="predict",
#             )

#     def _gen_dataloader(self, stage=None):
#         generator = None
#         if self.batch_seed is not None:
#             generator = torch.Generator()
#             generator = generator.manual_seed(self.batch_seed)

#         if stage == "train":
#             dataset = self.train_dataset
#             # Filter the dataset, if necessary
#             dataset.reroll()
#         elif stage == "eval":
#             dataset = self.eval_dataset
#         elif stage == "predict":
#             dataset = self.predict_dataset
#         else:
#             raise ValueError("Invalid stage")

#         batch_collator = OpenFoldBatchCollator()

#         dl = OpenFoldDataLoader(
#             dataset,
#             config=self.config,
#             stage=stage,
#             generator=generator,
#             batch_size=self.config.data_module.data_loaders.batch_size,
#             num_workers=self.config.data_module.data_loaders.num_workers,
#             collate_fn=batch_collator,
#         )

#         return dl

#     def train_dataloader(self):
#         return self._gen_dataloader("train")

#     def val_dataloader(self):
#         if self.eval_dataset is not None:
#             return self._gen_dataloader("eval")
#         return [] 

#     def predict_dataloader(self):
#         return self._gen_dataloader("predict")
