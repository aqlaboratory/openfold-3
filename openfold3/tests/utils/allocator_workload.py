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

"""OF3 training steps of varying size, for measuring CUDA allocator behaviour.

Allocator settings are process-global and fixed once CUDA allocates, so each
configuration must run in a fresh interpreter:

    python -m openfold3.tests.utils.allocator_workload --expandable-segments on \\
        --out metrics.json
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import torch

# Import through the CLI module, so CUDA is initialised exactly as in production
# before the allocator is configured.
from openfold3.run_openfold import _configure_torch_backend
from openfold3.tests.utils.cuda_memory import CudaMemoryMetrics, get_cuda_memory_metrics

# Varying sizes are what fragment the caching allocator; a single size would
# reuse the same blocks every step.
DEFAULT_N_TOKENS = (128, 256, 160, 320, 192, 320)


def run_training_steps(n_tokens: Sequence[int], n_msa: int) -> list[CudaMemoryMetrics]:
    """Forward, loss, backward and optimizer step for each crop size.

    Returns allocator metrics per step. Peaks are reset before each step but the
    cache is not emptied: blocks cached by earlier, differently sized steps are
    exactly what fragments the allocator.
    """
    from openfold3.core.loss.loss_module import OpenFold3Loss
    from openfold3.core.utils.precision_utils import OF3DeepSpeedPrecision
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from openfold3.tests.utils.data_utils import random_of3_features

    device = torch.device("cuda")
    torch.manual_seed(0)

    config = OF3ProjectEntry().get_model_config_with_presets()
    config.settings.blocks_per_ckpt = 1
    config.settings.ckpt_intermediate_steps = True
    # Same reduced model as TestOF3Model, to fit CI GPUs
    config.architecture.pairformer.no_blocks = 4
    config.architecture.diffusion_module.diffusion_transformer.no_blocks = 4
    config.architecture.loss_module.diffusion.chunk_size = 16

    model = OpenFold3AllAtom(config).to(device=device, dtype=torch.bfloat16)
    loss_fn = OpenFold3Loss(config=config.architecture.loss_module)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    precision = OF3DeepSpeedPrecision(precision="bf16-mixed")

    metrics = []
    for step, n_token in enumerate(n_tokens):
        torch.cuda.reset_peak_memory_stats(device)
        # Same data regardless of allocator settings
        torch.manual_seed(1000 + step)
        batch = random_of3_features(
            batch_size=1, n_token=n_token, n_msa=n_msa, n_templ=4
        )
        batch = precision.convert_input(batch)
        batch = tensor_tree_map(lambda t: t.to(device), batch)

        batch, outputs = model(batch=batch)
        loss = loss_fn(batch=batch, output=outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del batch, outputs, loss
        torch.cuda.synchronize(device)
        metrics.append(get_cuda_memory_metrics(device))

    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expandable-segments", choices=["on", "off"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--n-tokens",
        type=lambda s: [int(n) for n in s.split(",")],
        default=list(DEFAULT_N_TOKENS),
    )
    parser.add_argument("--n-msa", type=int, default=1024)
    args = parser.parse_args(argv)

    _configure_torch_backend(expandable_segments=args.expandable_segments == "on")

    steps = run_training_steps(n_tokens=args.n_tokens, n_msa=args.n_msa)
    segments = torch.cuda.memory._snapshot()["segments"]
    result = {
        "is_expandable": all(s["is_expandable"] for s in segments),
        "n_tokens": args.n_tokens,
        "steps": [dataclasses.asdict(m) for m in steps],
    }
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
