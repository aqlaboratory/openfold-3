"""
Converts an OpenFold3 checkpoint file to another model version.

Use it to hand a checkpoint written by this code to an older OpenFold3 release,
e.g. to resume a run on older code or to publish parameters for it:

    python scripts/dev/convert_checkpoint_version.py last.ckpt last-v2.0.0.ckpt \
        --version 2.0.0

Older versions are listed in openfold3/projects/of3_all_atom/checkpoint_migrations.py.
Loading needs no conversion: checkpoints from older versions are migrated as they
are loaded.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch

from openfold3.core.utils.checkpoint_loading_utils import load_checkpoint
from openfold3.projects.of3_all_atom.checkpoint_migrations import (
    VERSION_KEY,
    Version,
    as_version,
    migrate_checkpoint,
)


def find_versions(ckpt: dict[str, Any]) -> set[Version]:
    state_dicts = [
        ckpt,
        ckpt.get("state_dict"),
        ckpt.get("module"),
        (ckpt.get("ema") or {}).get("params"),
    ]
    return {
        as_version(v)
        for sd in state_dicts
        if isinstance(sd, dict)
        for k, v in sd.items()
        if isinstance(k, str) and k.endswith(VERSION_KEY)
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input_ckpt_path", type=Path)
    parser.add_argument("output_ckpt_path", type=Path)
    parser.add_argument(
        "--version", required=True, help="Target model version, e.g. 2.0.0"
    )
    args = parser.parse_args()

    target = as_version(args.version.split("."))
    ckpt = load_checkpoint(args.input_ckpt_path)
    before = find_versions(ckpt)
    migrate_checkpoint(ckpt, target)
    after = find_versions(ckpt)
    if after != {target}:
        raise ValueError(f"No migration path from {before} to {target}")

    torch.save(ckpt, args.output_ckpt_path)
    print(f"Converted {args.input_ckpt_path} {before}")
    print(f"       to {args.output_ckpt_path} {after}")


if __name__ == "__main__":
    main()
