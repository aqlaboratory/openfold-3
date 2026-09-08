"""Run the named-bond glycan example through inference and output writing.

This smoke test checks that the query produces finite coordinates with its
attachment atoms present. Bond distances and pose accuracy are not asserted.
Requires an accelerator and model weights, as do the other inference tests.
"""

from pathlib import Path

import numpy as np
import pytest

import openfold3
from openfold3.core.metrics.alignment import Structure
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.tests.inference.helpers import predicted_structure_cifs, run_inference
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

EXAMPLE = (
    Path(openfold3.__file__).parent.parent
    / "examples/example_inference_inputs/query_asn_two_sugar_glycan.json"
)


@pytest.mark.slow
@pytest.mark.skipif(not EXAMPLE.is_file(), reason="Requires source checkout examples")
@skip_unless_accelerator_available()
def test_covalent_glycan_inference(tmp_path):
    query_set = InferenceQuerySet.from_json(EXAMPLE)
    run_inference(
        query_set,
        tmp_path,
        use_msa_server=False,
        use_templates=False,
        num_diffusion_samples=1,
    )
    samples = predicted_structure_cifs(tmp_path, "asn_two_sugar_glycan")
    assert len(samples) == 1, "Expected one completed glycan prediction"
    atoms = Structure.from_cif(samples[0]).atom_array
    assert np.isfinite(atoms.coord).all(), "Prediction contains non-finite coordinates"
    for chain, name in [("A", "ND2"), ("G", "C1")]:
        endpoint = (
            (atoms.chain_id == chain) & (atoms.res_id == 1) & (atoms.atom_name == name)
        )
        assert endpoint.sum() == 1, (
            f"Missing or ambiguous attachment atom {chain}:1:{name}"
        )
