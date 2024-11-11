# Copyright 2021 AlQuraishi Laboratory
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
import importlib
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.utils.checkpoint

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed
    from deepspeed.runtime.activation_checkpointing.checkpointing import (
        non_reentrant_checkpoint as ds_non_reentrant_checkpoint,
    )


BLOCK_ARG = Any
BLOCK_ARGS = Sequence[BLOCK_ARG]


def get_checkpoint_fn(
    deepspeed_is_configured: bool, use_reentrant: Optional[bool] = None
):
    if deepspeed_is_configured:
        if use_reentrant is False:
            checkpoint = ds_non_reentrant_checkpoint
        else:
            checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint

    return checkpoint


@torch.jit.ignore
def checkpoint_blocks(
    blocks: List[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: Optional[int],
    use_reentrant: Optional[bool] = None,
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        blocks_per_ckpt:
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
        use_reentrant:
            Whether to use reentrant checkpointing. If set, torch checkpointing
            will be used.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    deepspeed_is_configured = (
        deepspeed_is_installed and deepspeed.checkpointing.is_configured()
    )

    checkpoint = get_checkpoint_fn(
        deepspeed_is_configured=deepspeed_is_configured, use_reentrant=use_reentrant
    )

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt

        if deepspeed_is_configured:
            args = checkpoint(chunker(s, e), *args)
        else:
            args = checkpoint(chunker(s, e), *args, use_reentrant=use_reentrant)

        args = wrap(args)

    return args


@torch.jit.ignore
def checkpoint_section(
    fn: Callable,
    args: BLOCK_ARGS,
    apply_ckpt: Optional[bool] = True,
    use_reentrant: Optional[bool] = None,
):
    """
    Apply checkpointing to a single function.

    Args:
        fn:
            Function to checkpoint
        args:
            Tuple of arguments for the first block.
        apply_ckpt:
            Whether to apply checkpointing. If None, no checkpointing
            is performed.
        use_reentrant:
            Whether to use reentrant checkpointing. If set, torch checkpointing
            will be used.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(fn, a):
        return fn(*a)

    # Avoids mishaps when the function takes just one argument
    args = wrap(args)

    if not apply_ckpt or not torch.is_grad_enabled():
        return exec(fn, args)

    deepspeed_is_configured = (
        deepspeed_is_installed and deepspeed.checkpointing.is_configured()
    )

    checkpoint = get_checkpoint_fn(
        deepspeed_is_configured=deepspeed_is_configured, use_reentrant=use_reentrant
    )

    if deepspeed_is_configured:
        args = checkpoint(fn, *args)
    else:
        args = checkpoint(fn, *args, use_reentrant=use_reentrant)

    return args
