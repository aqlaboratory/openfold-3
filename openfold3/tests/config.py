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

import ml_collections as mlc

monomer_consts = mlc.ConfigDict(
    {
        "model_name": "af2_monomer",
        "model_preset": "model_1_ptm",
        "is_multimer": False,  # monomer: False, multimer: True
        "chunk_size": 4,
        "batch_size": 2,
        "n_res": 22,
        "n_seq": 13,
        "n_templ": 3,
        "n_extra": 17,
        "n_heads_extra_msa": 8,
        "inf": 1e5,
        "eps": 5e-4,
        # For compatibility with DeepMind's pretrained weights, it's easiest for
        # everyone if these take their real values.
        "c_m": 256,
        "c_z": 128,
        "c_s": 384,
        "c_t": 64,
        "c_e": 64,
        "msa_logits": 23,  # monomer: 23, multimer: 22
        "template_mmcif_dir": None,  # Set for test_multimer_datamodule
    }
)

multimer_consts = mlc.ConfigDict(
    {
        "model_name": "af2_multimer",
        "model_preset": "model_1_multimer_v3",
        "is_multimer": True,  # monomer: False, multimer: True
        "chunk_size": 4,
        "batch_size": 2,
        "n_res": 22,
        "n_seq": 13,
        "n_templ": 3,
        "n_extra": 17,
        "n_heads_extra_msa": 8,
        "inf": 1e5,
        "eps": 5e-4,
        # For compatibility with DeepMind's pretrained weights, it's easiest for
        # everyone if these take their real values.
        "c_m": 256,
        "c_z": 128,
        "c_s": 384,
        "c_t": 64,
        "c_e": 64,
        "msa_logits": 22,  # monomer: 23, multimer: 22
        "template_mmcif_dir": None,  # Set for test_multimer_datamodule
    }
)

consts = monomer_consts

config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "masked_msa": {
                    "profile_prob": 0.1,
                    "same_prob": 0.1,
                    "uniform_prob": 0.1,
                },
            }
        }
    }
)
