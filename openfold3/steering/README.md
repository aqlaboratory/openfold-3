# Chemical steering

Gradient guidance of the diffusion sampler toward valid ligand chemistry.
Full design: see the project's steering design doc; this file covers what's
actually in the directory and where it came from.

## Architecture

Three phases, and the split between them is the central decision:

```
FEATURIZATION (CPU, once per query, RDKit)
    ProcessedReferenceMolecule.mol  +  in_crop_mask
        --> featurization.py --> RestraintSet per term  --> SteeringContext

SAMPLING LOOP (GPU, every denoising step, tensors only)
    x0_hat = diffusion_module(...)
    update = steering.on_denoised(x0_hat, StepState(...))
    x0_hat = x0_hat + update.delta
```

**RDKit never enters the sampling loop.** Constraint extraction is chemistry
and runs once on CPU (`featurization.py`); the loop sees index tensors and
bounds (`engine.py`, `potentials.py`). Weights and intervals can be scheduled
over the trajectory (`schedules.py`); bounds cannot without re-extraction.

## Status

One restraint family, `DistanceBoundsPotential` (`distance_bounds_potential`
in a runner yaml), wired end to end: a run
turns steering on with `dataset_config_kwargs.steering.enabled` (see
`examples/example_runner_yamls/chemical_steering.yml`), the inference dataset
emits restraints, and the hook in `diffusion_module.py` applies them during
the primary rollout. The pocket-refinement pass is deliberately left
unsteered.

`openfold3/tests/inference/test_chemical_steering.py` is the end-to-end
check: FKBP12 with rapamycin, run with steering on and off, measuring the
predicted ligand's internal geometry. Marked slow; needs an accelerator and
weights.

Follow-on work: the remaining restraint families (VDW overlap, chiral atom,
stereo bond, planar bond), the science tests that go with them, and
Feynman-Kac resampling.

Module map:

| file | role |
|---|---|
| `types.py` | `RestraintSet`, `SteeringContext`, `StepState`, `SteeringUpdate` |
| `defaults.py` | derived parameter table; referenced, never restated |
| `schedules.py` | time-varying per-term weights |
| `potentials.py` | `Potential` ABC, the registry, `DistanceBoundsPotential` |
| `engine.py` | `ChemicalSteering.on_denoised`, the guidance loop |
| `featurization.py` | RDKit chemistry and `maybe_create_steering_features` (data side; the only module importing RDKit) |
| `batch_features.py` | the batch contract: flatten a context out, rebuild it in the sampler (shared by both sides) |
| `config.py` | `SteeringSettings` |

## Provenance policy

The boundary is drawn on **derivation, not topic**, and it runs *through*
this directory rather than around it:

- **Derived** — `potentials.py`, `engine.py`, the extraction rules in
  `featurization.py`,
  `schedules.py`, `defaults.py`, and the tests that were ported alongside
  them. These adapt one or more of the three prior projects below.
- **Original OpenFold3, deriving from nothing** — `types.py`, `config.py`,
  `batch_features.py`, and `maybe_create_steering_features` in
  `featurization.py`. These are plumbing: the settings model,
  the batch wire format, and the featurization entry point. They live here
  for cohesion (steering is one feature, kept in one place) rather than
  because they carry upstream content.

Outside this directory, code merely *invokes* steering rather than
implementing it and carries no derived content.

See `THIRD_PARTY_NOTICES.md` for full license texts and a per-subject
breakdown. Summary:

- **Boltz** (MIT) — the flat-bottom formulation, the default weights and
  buffers (`defaults.py`), and the RDKit extraction rules
  (`featurization.py`).
- **Protenix** (Apache 2.0) — the potential-class registry structure
  (`CLASS_REGISTRY` / `register` in `potentials.py`).
- **foldsteer** (Apache 2.0) — the engine (`engine.py`) and schedule types
  (`schedules.py`), and the broadcast-regression test pattern.

**Derived defaults live in `defaults.py` and nowhere else.** Config models
outside this directory must reference them, never restate the numeric
values — that duplication is exactly how a derived constant ends up
uncredited outside the directory boundary.

Credit: constraint extraction and the acceptance test suite this package's
tests are ported from originate with Peter Obi (OpenFold3 PR #385); the
engine, registry, and schedule design originate with Etowah Adams
(foldsteer). See `THIRD_PARTY_NOTICES.md` and this repository's
`CITATION.cff`.
