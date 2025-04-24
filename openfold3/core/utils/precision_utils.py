# Copyright 2022 AlQuraishi Laboratory
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

from typing import Any

import torch
from lightning_fabric.plugins.precision.deepspeed import _PRECISION_INPUT
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor
from lightning_utilities import apply_to_collection
from pytorch_lightning.plugins.precision.deepspeed import DeepSpeedPrecision


def is_fp16_enabled():
    # Autocast world
    fp16_enabled = (
        torch.cuda.is_available() and torch.get_autocast_dtype("cuda") == torch.float16
    )
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()

    return fp16_enabled


class OF3DeepSpeedPrecision(DeepSpeedPrecision):
    """Precision plugin to selectively convert inputs to the desired precision."""

    def __init__(self, precision: _PRECISION_INPUT) -> None:
        super().__init__(precision=precision)

    def convert_input(self, data: Any) -> Any:
        """
        Converts input data to the desired precision.
        The ground truth and reference conformer features will not be cast
        to a lower precision in order to avoid truncating atom coordinates.

        Args:
            data: Input feature dictionary

        Returns:
            data: Converted input feature dictionary with the desired precision
        """
        ground_truth = data.pop("ground_truth", None)
        ref_conformer_feats = {k: v for k, v in data.items() if k.startswith("ref_")}

        data = apply_to_collection(
            data,
            function=_convert_fp_tensor,
            dtype=torch.Tensor,
            dst_type=self._desired_dtype,
        )

        data.update(ref_conformer_feats)

        if ground_truth is not None:
            data["ground_truth"] = ground_truth

        return data
