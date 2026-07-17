#!/usr/bin/env bash
#
# Download (or reuse) a small PDB training subset, then launch a training run
# on it via run_openfold. Any extra arguments are passed through to
# `run_openfold train` (e.g. --seed, --data-seed).
#
# Usage:
#   ./train_on_subset.sh
#   ./train_on_subset.sh --seed 1234
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER_YAML="${SCRIPT_DIR}/train_pdb_subset.yaml"

echo "Downloading PDB training subset (train=32, val=16 by default)..."
pixi run --manifest-path "${REPO_ROOT}/pixi.toml" -e openfold3-cuda12 \
  python "${SCRIPT_DIR}/download_subset.py"

echo "Starting training run with ${RUNNER_YAML} ..."
pixi run --manifest-path "${REPO_ROOT}/pixi.toml" -e openfold3-cuda12 \
  run_openfold train --runner-yaml="${RUNNER_YAML}" "$@"
