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

"""Steering features as the inference dataset actually emits them.

The package's own tests exercise `maybe_create_steering_features` directly; these
go through `InferenceDataset.create_all_features`, which is the only thing
that proves a run reaches the sampler with restraints. Without them, deleting
the call in `create_all_features` leaves every other steering test green while
steering silently never happens.

Ethanol is the molecule under test because its restraint set is small enough
to write out by hand: two bonds and the one 1-3 pair they share.
"""

from __future__ import annotations

import numpy as np
import pytest

from openfold3.core.data.framework.single_datasets.inference import InferenceDataset
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.dataset_configs import InferenceJobConfig
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.steering.batch_features import (
    STEERING_ENABLED_KEY,
    STEERING_N_ATOMS_KEY,
    STEERING_NUM_GD_STEPS_KEY,
    term_key,
)
from openfold3.steering.config import SteeringSettings

_TERM = "distance_bounds_potential"

# A protein chain precedes the ligand so the ligand does not start at atom 0:
# restraints are built against RDKit-local indices, and the global offset is
# exactly what a wiring mistake would get wrong.
_PROTEIN = {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "GAGA"}
_ETHANOL = {"molecule_type": "ligand", "chain_ids": ["L"], "smiles": "CCO"}

# Ethanol, heavy atoms only: both bonds plus the 1-3 pair across them.
_EXPECTED_PAIRS = {("C1", "C2"), ("C2", "O1"), ("C1", "O1")}


def _dataset(chains: list[dict], settings: SteeringSettings) -> InferenceDataset:
    query_set = InferenceQuerySet.model_validate(
        {"queries": {"query": {"chains": chains}}}
    )
    return InferenceDataset(
        dataset_config=InferenceJobConfig(
            query_set=query_set,
            template_preprocessor_settings=TemplatePreprocessorSettings(),
            steering=settings,
        )
    )


def _features(chains: list[dict], settings: SteeringSettings) -> dict:
    dataset = _dataset(chains, settings)
    return dataset.create_all_features(dataset.query_cache["query"])


@pytest.fixture
def ethanol_features() -> dict:
    return _features([_PROTEIN, _ETHANOL], SteeringSettings(enabled=True))


def test_enabled_steering_emits_exactly_the_expected_feature_keys(ethanol_features):
    """The key set is the contract with `prepare_steering`; a missing key
    raises there rather than degrading, and an extra one would be dropped
    silently."""
    emitted = {key for key in ethanol_features if key.startswith("steering_")}

    assert emitted == {
        STEERING_ENABLED_KEY,
        STEERING_NUM_GD_STEPS_KEY,
        STEERING_N_ATOMS_KEY,
        term_key(_TERM, "atom_index"),
        term_key(_TERM, "lower"),
        term_key(_TERM, "upper"),
        term_key(_TERM, "count"),
        term_key(_TERM, "weight"),
        term_key(_TERM, "interval"),
    }


def test_emitted_features_agree_with_the_structure_they_describe(ethanol_features):
    settings = SteeringSettings(enabled=True)
    atom_array = ethanol_features["atom_array"]
    count = int(ethanol_features[term_key(_TERM, "count")].item())
    atom_index = ethanol_features[term_key(_TERM, "atom_index")]

    assert bool(ethanol_features[STEERING_ENABLED_KEY].item())
    assert int(ethanol_features[STEERING_N_ATOMS_KEY].item()) == len(atom_array)
    assert (
        int(ethanol_features[STEERING_NUM_GD_STEPS_KEY].item()) == settings.num_gd_steps
    )
    assert float(ethanol_features[term_key(_TERM, "weight")].item()) == pytest.approx(
        settings.terms[_TERM].weight
    )
    # `count` is the shape authority downstream -- the collator reshapes these
    # tensors, so orientation is never inferred from `atom_index.shape`.
    assert atom_index.numel() == count * 2
    assert ethanol_features[term_key(_TERM, "lower")].numel() == count
    assert ethanol_features[term_key(_TERM, "upper")].numel() == count


def test_ethanol_generates_its_three_distance_constraints(ethanol_features):
    """Ethanol's heavy-atom restraints, enumerated: C1-C2, C2-O1, and the
    1-3 pair C1...O1. Identified by atom name, so this pins that the global
    offset of the ligand -- 18, not 0, behind a four-residue protein chain --
    is applied to every index."""
    atom_array = ethanol_features["atom_array"]
    atom_index = ethanol_features[term_key(_TERM, "atom_index")]

    ligand_atoms = np.flatnonzero(atom_array.molecule_type_id == MoleculeType.LIGAND)
    assert ligand_atoms.min() > 0, "ligand must not start at atom 0 for this test"

    named = {
        frozenset((str(atom_array.atom_name[i]), str(atom_array.atom_name[j])))
        for i, j in atom_index.tolist()
    }
    assert named == {frozenset(pair) for pair in _EXPECTED_PAIRS}

    referenced = set(atom_index.flatten().tolist())
    assert referenced <= set(ligand_atoms.tolist())


def test_ethanol_bounds_bracket_the_real_bond_lengths(ethanol_features):
    """Bounds are physical, not placeholders: each bond's window contains its
    textbook length, and the 1-3 pair's window sits above both."""
    atom_array = ethanol_features["atom_array"]
    atom_index = ethanol_features[term_key(_TERM, "atom_index")]
    lower = ethanol_features[term_key(_TERM, "lower")]
    upper = ethanol_features[term_key(_TERM, "upper")]

    windows = {
        frozenset((str(atom_array.atom_name[i]), str(atom_array.atom_name[j]))): (
            float(lower[k]),
            float(upper[k]),
        )
        for k, (i, j) in enumerate(atom_index.tolist())
    }

    for pair, expected in (
        (("C1", "C2"), 1.52),  # C-C single bond
        (("C2", "O1"), 1.43),  # C-O single bond
        (("C1", "O1"), 2.37),  # 1-3 separation across the C-C-O angle
    ):
        low, high = windows[frozenset(pair)]
        assert low < expected < high, (
            f"{pair} window [{low}, {high}] excludes {expected}"
        )


def test_steering_features_are_absent_when_steering_is_disabled():
    """Off by default, and "off" means no keys at all: the sampler probes for
    the enable key with `.get`, so an absent key is what makes a disabled run
    bit-identical to one without steering."""
    features = _features([_PROTEIN, _ETHANOL], SteeringSettings())

    assert not [key for key in features if key.startswith("steering_")]


def test_a_ligandless_query_emits_no_steering_features():
    """Enabled but nothing to steer: a protein-only query must take the same
    no-key path rather than emitting an empty restraint set."""
    features = _features([_PROTEIN], SteeringSettings(enabled=True))

    assert not [key for key in features if key.startswith("steering_")]
