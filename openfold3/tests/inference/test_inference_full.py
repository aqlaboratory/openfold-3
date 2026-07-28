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

"""Smoke tests for inference: small queries run end-to-end.

``test_inference_writes_outputs`` runs every case in ``CASES`` against every combination
of ``use_msa_server`` and ``use_templates``, checking the expected output files are
written for each query the case contains. Cases mix in-memory query sets with the query
JSONs under ``examples/example_inference_inputs``, so adding either kind is one row —
and costs four inference runs. See ``test_templates.py`` for the functional check that a
supplied template actually steers the prediction.

These require an accelerator (CUDA, ROCm or MPS) and downloaded model weights; they skip
otherwise.

Run with:
    pytest openfold3/tests/inference/test_inference_full.py
"""

import logging
from functools import partial
from pathlib import Path

import pytest

import openfold3
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.tests.inference.helpers import run_inference
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

#: Repo-root ``examples/`` — present in a source checkout, but not shipped in the wheel
#: (``packages.find`` only picks up ``openfold3*``), so file-backed cases must tolerate
#: its absence when the suite runs from an install.
EXAMPLES_DIR = (
    Path(openfold3.__file__).parent.parent / "examples" / "example_inference_inputs"
)

requires_examples = pytest.mark.skipif(
    not EXAMPLES_DIR.is_dir(), reason=f"No examples directory at {EXAMPLES_DIR}"
)

#: Seed the outputs are written under. Comes from ``ExperimentSettings.seeds`` (default
#: ``[42]``), which the runner yaml in :func:`run_inference` does not override — *not*
#: from ``InferenceQuerySet.seeds``, which the runner ignores.
SEED = 42

PROTEIN_CHAIN = {
    "molecule_type": "protein",
    "chain_ids": ["A", "B"],
    "sequence": "XRMKQLEDKVEELLSKNYHLENEVARLKKLVGER",
}

LIGAND_CHAIN = {
    "molecule_type": "ligand",
    "chain_ids": ["C"],
    "smiles": "c1ccccc1O",
}


def _inline_query_set(*chains: dict) -> InferenceQuerySet:
    """Build a single-query set named ``query1`` from raw chain dicts."""
    return InferenceQuerySet.model_validate(
        {"queries": {"query1": {"chains": list(chains)}}}
    )


def _example_query_set(filename: str) -> InferenceQuerySet:
    """Load an example query JSON through the same loader ``run_openfold`` uses."""
    return InferenceQuerySet.from_json(EXAMPLES_DIR / filename)


#: Each case builds an :class:`InferenceQuerySet`, either in memory or from a file in
#: ``examples/example_inference_inputs``. Construction is deferred behind a callable so
#: that a missing examples directory skips the case instead of breaking collection.
#: Expected outputs are derived from the query names, so a case may hold any number of
#: queries — adding another example is one row.
CASES = [
    pytest.param(partial(_inline_query_set, PROTEIN_CHAIN), id="protein_only"),
    pytest.param(
        partial(_inline_query_set, PROTEIN_CHAIN, LIGAND_CHAIN),
        id="protein_and_ligand",
    ),
    pytest.param(
        partial(_example_query_set, "query_protein_ligand_multiple.json"),
        id="protein_ligand_multiple",
        marks=requires_examples,
    ),
]


@skip_unless_accelerator_available()
@pytest.mark.parametrize("build_query_set", CASES)
@pytest.mark.parametrize(
    "use_templates", [False, True], ids=["no_templates", "templates"]
)
@pytest.mark.parametrize("use_msa_server", [False, True], ids=["no_msa", "msa"])
def test_inference_writes_outputs(
    build_query_set, use_msa_server, use_templates, tmp_path
):
    """Every query in the set writes the expected per-sample files, in every mode.

    The two feature flags are independent branches of ``prepare_data``: without the MSA
    server the query sequence is used single-sequence, and template preprocessing runs
    on its own flag. Stacking the three parametrize marks gives the full cartesian
    product, so each query set is exercised in all four modes.
    """
    query_set = build_query_set()
    run_inference(
        query_set,
        tmp_path,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
        num_diffusion_samples=1,
        # Isolate the template cache: without this it lands in a persistent /tmp dir
        # shared across runs, and the template-enabled cases here would see each
        # other's leftovers.
        template_output_dir=tmp_path / "template_data",
    )
    logger.info("Checking output contents at %s", tmp_path)
    for query_name in query_set.queries:
        seed_dir = tmp_path / query_name / f"seed_{SEED}"
        stem = f"{query_name}_seed_{SEED}_sample_1"
        expected_files = [
            f"{stem}_confidences.json",
            f"{stem}_confidences_aggregated.json",
            f"{stem}_model.cif",
            "timing.json",
        ]
        for name in expected_files:
            assert (seed_dir / name).exists(), (
                f"Expected output file not found: {seed_dir / name}"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
