# Copyright 2026 AlQuraishi Laboratory
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

import logging

import pytorch_lightning as pl
from pytorch_lightning.overrides.distributed import _sync_module_states
from pytorch_lightning.strategies import DDPStrategy

logger = logging.getLogger(__name__)


class Rank0BroadcastStrategy(DDPStrategy):
    """
    DDP without wrapping the model in ``DistributedDataParallel``.

    Per-sample gradient clipping means every backward pass runs inside
    ``no_sync()`` and gradients are reduced by hand in
    ``PerSampleGradManager._sync_and_average_grads``. That leaves DDP's
    ``Reducer`` permanently inert:

    * Lightning clears ``require_backward_grad_sync`` for manual optimization
      (``_DDPForwardRedirection.on_after_inner_forward``), and the backward
      itself runs under ``no_sync()``, so ``prepare_for_backward`` is never
      called from either ``DistributedDataParallel.forward`` or
      ``DDPStrategy.pre_backward``.
    * Without ``prepare_for_backward``, ``Reducer::autograd_hook`` returns
      early on every gradient, so no gradient is ever copied into a bucket.

    The wrapper still allocates the buckets when it is built, though: one flat
    buffer per bucket, together covering every gradient, i.e. a full copy of
    the parameters in their own dtype that is never read or written. Skipping
    the wrapper gives that copy back.

    The only thing the wrapper did that is still needed is the rank-0 broadcast
    of parameters and buffers at setup, which is done explicitly here.
    """

    """
    Broadcasts parameters from rank 0, but does not wrap model in DDP.
    """
    def configure_ddp(self) -> None:
        assert isinstance(self.model, pl.LightningModule)
        _sync_module_states(self.model)
