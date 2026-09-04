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

"""The batch wire format: flatten, survive the collator, rebuild.

The shape-mangling cases here are not hypothetical -- the collator applies
`pad_sequence(...).squeeze(-1)` to every feature and the model then
`unsqueeze(1)`s every leaf, so these are the shapes that actually reach the
sampler.
"""

from __future__ import annotations

import pytest
import torch

from openfold3.steering.batch_features import (
    STEERING_ENABLED_KEY,
    STEERING_N_ATOMS_KEY,
    STEERING_NUM_GD_STEPS_KEY,
    context_to_features,
    prepare_steering,
    steering_enabled,
    term_key,
)
from openfold3.steering.config import SteeringSettings
from openfold3.steering.featurization import (
    build_context,
    maybe_create_steering_features,
)
from openfold3.steering.potentials import (
    DistanceBoundsPotential,
)
from openfold3.steering.tests._structures import (
    LAYOUTS,
    collate,
    reference_coords_by_atom_name,
    structure_for,
)
from openfold3.steering.types import RestraintSet, SteeringContext

_TERM = "distance_bounds_potential"


def _context(n_restraints: int = 2, n_atoms: int = 5) -> SteeringContext:
    atom_index = torch.tensor(
        [[i, i + 1] for i in range(n_restraints)], dtype=torch.int64
    ).reshape(n_restraints, 2)
    return SteeringContext(
        restraints={
            _TERM: RestraintSet(
                atom_index=atom_index,
                lower=torch.full((n_restraints,), 1.0),
                upper=torch.full((n_restraints,), 1.5),
            )
        },
        n_atoms=n_atoms,
    )


def _enabled_settings(**kwargs) -> SteeringSettings:
    return SteeringSettings.model_validate({"enabled": True, **kwargs})


def test_feature_keys_are_derived_from_the_registry_key():
    assert term_key(_TERM, "lower") == "steering_distance_bounds_potential_lower"


def test_a_newly_registered_potential_gets_keys_with_no_bookkeeping(
    register_throwaway, isolated_registry
):
    """Feature names are derived, not declared: registering a potential is
    the only step: nothing in features.py has to be updated for its keys to
    exist, so a new term cannot be silently left out of the wire format."""
    register_throwaway("torsion_angle_potential", arity=4)

    assert term_key("torsion_angle_potential", "count") == (
        "steering_torsion_angle_potential_count"
    )
    assert "torsion_angle_potential" in isolated_registry


def test_feature_keys_are_distinct_across_the_registry(
    register_throwaway, isolated_registry
):
    """Two potentials whose derived names collided would silently share a
    slot in the batch."""
    register_throwaway("improper_dihedral_potential", arity=4)

    keys = {term_key(name, "atom_index") for name in isolated_registry}
    assert len(keys) == len(isolated_registry), "feature keys must be distinct"


def test_context_to_features_emits_the_expected_key_set():
    features = context_to_features(_context(), _enabled_settings())

    assert set(features) == {
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
    assert features[term_key(_TERM, "count")].tolist() == [2]
    assert features[term_key(_TERM, "atom_index")].dtype == torch.int64
    assert features[term_key(_TERM, "lower")].dtype == torch.float32


def test_context_to_features_is_empty_when_no_term_is_active():
    """No keys at all is what makes disabled steering a structural no-op."""
    settings = _enabled_settings(terms={_TERM: {"enabled": False}})
    assert context_to_features(_context(), settings) == {}


def test_context_to_features_is_empty_when_there_are_no_restraints():
    empty = SteeringContext(
        restraints={
            _TERM: RestraintSet(
                atom_index=torch.empty((0, 2), dtype=torch.int64),
                lower=torch.empty((0,)),
                upper=torch.empty((0,)),
            )
        },
        n_atoms=5,
    )
    assert context_to_features(empty, _enabled_settings()) == {}


def test_steering_enabled_is_false_without_features():
    assert steering_enabled({}) is False
    assert steering_enabled({STEERING_ENABLED_KEY: torch.tensor([False])}) is False
    assert steering_enabled({STEERING_ENABLED_KEY: torch.tensor([True])}) is True


def test_prepare_steering_round_trips_a_context():
    ctx = _context()
    settings = _enabled_settings(num_gd_steps=7)
    features = context_to_features(ctx, settings)

    prepared = prepare_steering(features, torch.ones(1, ctx.n_atoms))

    assert prepared is not None
    assert prepared.engine.num_gd_steps == 7
    rebuilt = prepared.ctx.restraints[_TERM]
    original = ctx.restraints[_TERM]
    torch.testing.assert_close(rebuilt.atom_index, original.atom_index)
    torch.testing.assert_close(rebuilt.lower, original.lower)
    torch.testing.assert_close(rebuilt.upper, original.upper)
    assert prepared.ctx.n_atoms == ctx.n_atoms

    term = prepared.engine.terms[_TERM]
    assert term.weight.at(0.5) == pytest.approx(settings.terms[_TERM].weight)
    assert term.interval == settings.terms[_TERM].interval


def test_prepare_steering_returns_none_without_features():
    assert prepare_steering({}, torch.ones(1, 5)) is None


@pytest.mark.parametrize("n_restraints", [1, 2, 5])
def test_prepare_steering_survives_the_real_collation_path(n_restraints: int):
    """n_restraints=1 is the interesting case: `[n]` bounds collapse to `[1]`
    under squeeze(-1), so nothing may infer shape from the tensor itself."""
    ctx = _context(n_restraints=n_restraints, n_atoms=n_restraints + 1)
    features = collate(context_to_features(ctx, _enabled_settings()))

    prepared = prepare_steering(features, torch.ones(1, ctx.n_atoms))

    assert prepared is not None
    rebuilt = prepared.ctx.restraints[_TERM]
    assert rebuilt.atom_index.shape == (n_restraints, 2)
    torch.testing.assert_close(rebuilt.atom_index, ctx.restraints[_TERM].atom_index)
    torch.testing.assert_close(rebuilt.lower, ctx.restraints[_TERM].lower)


def test_prepare_steering_rejects_multi_query_batches():
    features = context_to_features(_context(), _enabled_settings())
    with pytest.raises(ValueError, match="one query per model batch"):
        prepare_steering(features, torch.ones(2, 5))


def test_prepare_steering_rejects_an_atom_axis_mismatch():
    features = context_to_features(_context(n_atoms=5), _enabled_settings())
    with pytest.raises(ValueError, match="built for 5 atoms"):
        prepare_steering(features, torch.ones(1, 9))


def test_prepare_steering_rejects_a_missing_feature():
    features = context_to_features(_context(), _enabled_settings())
    del features[term_key(_TERM, "lower")]
    with pytest.raises(ValueError, match="lower.* is missing"):
        prepare_steering(features, torch.ones(1, 5))


def test_prepare_steering_rejects_out_of_range_indices():
    features = context_to_features(_context(n_atoms=5), _enabled_settings())
    features[term_key(_TERM, "atom_index")] = torch.tensor(
        [[0, 1], [2, 99]], dtype=torch.int64
    )
    with pytest.raises(ValueError, match="outside"):
        prepare_steering(features, torch.ones(1, 5))


def test_prepare_steering_rejects_a_count_mismatch():
    features = context_to_features(_context(), _enabled_settings())
    features[term_key(_TERM, "count")] = torch.tensor([3], dtype=torch.long)
    with pytest.raises(ValueError, match="expected 6 for 3 restraint"):
        prepare_steering(features, torch.ones(1, 5))


def test_prepare_steering_rejects_a_float_index_tensor():
    features = context_to_features(_context(), _enabled_settings())
    features[term_key(_TERM, "atom_index")] = features[
        term_key(_TERM, "atom_index")
    ].float()
    with pytest.raises(ValueError, match="must be an integer tensor"):
        prepare_steering(features, torch.ones(1, 5))


# ---------------------------------------------------------------------------
# The featurization entry point: a real molecule all the way to batch tensors.
# ---------------------------------------------------------------------------


def _ligand_structure(smiles: str = "C[C@H](O)C(=O)O"):
    return structure_for(
        [{"molecule_type": "ligand", "chain_ids": ["L"], "smiles": smiles}]
    )


def _enabled() -> SteeringSettings:
    return SteeringSettings(enabled=True)


def test_features_are_empty_when_steering_is_disabled():
    structure = _ligand_structure()
    features = maybe_create_steering_features(
        structure.atom_array, structure.processed_reference_mols, SteeringSettings()
    )
    assert features == {}


def test_features_are_empty_when_settings_are_omitted():
    """The default is off, so an unconfigured run emits nothing."""
    structure = _ligand_structure()
    assert (
        maybe_create_steering_features(
            structure.atom_array, structure.processed_reference_mols
        )
        == {}
    )


def test_features_are_empty_when_every_term_is_disabled():
    structure = _ligand_structure()
    settings = SteeringSettings.model_validate(
        {"enabled": True, "terms": {_TERM: {"enabled": False}}}
    )
    assert (
        maybe_create_steering_features(
            structure.atom_array, structure.processed_reference_mols, settings
        )
        == {}
    )


def test_features_are_empty_for_a_query_with_no_ligand():
    """A protein-only query yields no restraints, so steering no-ops on it
    even in a steered run."""
    structure = structure_for(
        [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "GAGA"}]
    )
    features = maybe_create_steering_features(
        structure.atom_array, structure.processed_reference_mols, _enabled()
    )
    assert features == {}


def test_features_for_a_ligand_query_carry_restraints():
    structure = _ligand_structure()
    features = maybe_create_steering_features(
        structure.atom_array, structure.processed_reference_mols, _enabled()
    )

    assert bool(features[STEERING_ENABLED_KEY].item()) is True
    assert int(features[STEERING_N_ATOMS_KEY].item()) == len(structure.atom_array)
    count = int(features[term_key(_TERM, "count")].item())
    assert count > 0
    assert features[term_key(_TERM, "atom_index")].shape == (count, 2)
    assert features[term_key(_TERM, "lower")].shape == (count,)


def test_featurization_round_trips_through_prepare_steering():
    """What the sampler rebuilds must equal what extraction produced.

    Narrow by design: this pins the flatten/rebuild step only. It cannot say
    the restraints are *correct*, because the value it compares against comes
    from the same build_context call maybe_create_steering_features makes
    internally. The end-to-end check is the test below.
    """
    structure = _ligand_structure()
    settings = _enabled()
    features = maybe_create_steering_features(
        structure.atom_array, structure.processed_reference_mols, settings
    )

    prepared = prepare_steering(features, torch.ones(1, len(structure.atom_array)))
    assert prepared is not None

    expected = build_context(
        structure.atom_array,
        structure.processed_reference_mols,
        n_atoms=len(structure.atom_array),
    ).restraints[_TERM]
    rebuilt = prepared.ctx.restraints[_TERM]

    torch.testing.assert_close(rebuilt.atom_index, expected.atom_index)
    torch.testing.assert_close(rebuilt.lower, expected.lower)
    torch.testing.assert_close(rebuilt.upper, expected.upper)
    assert prepared.engine.num_gd_steps == settings.num_gd_steps


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_restraints_survive_the_whole_path_from_molecule_to_sampler(layout: str):
    """Molecule -> features -> collator -> sampler-side constraints.

    Every step the real pipeline takes, ending in an assertion that does not
    reuse any of the machinery under test: the reference conformer each
    molecule's bounds were derived from is placed by *atom name*, and must
    satisfy the restraints the sampler ends up holding. A mangled shape, a
    dropped term, or a mis-mapped atom index all show up as non-zero energy.
    """
    structure = structure_for(LAYOUTS[layout])
    features = maybe_create_steering_features(
        structure.atom_array, structure.processed_reference_mols, _enabled()
    )
    assert features, "a steered run with a ligand must emit features"

    prepared = prepare_steering(
        collate(features), torch.ones(1, len(structure.atom_array))
    )
    assert prepared is not None

    coords = reference_coords_by_atom_name(structure)
    energy, _ = DistanceBoundsPotential().energy_and_gradient(
        coords, prepared.ctx.restraints[_TERM], 1.0
    )
    assert float(energy) == pytest.approx(0.0, abs=1e-6)
