# Third-Party Notices

This directory (`openfold3/steering/`) adapts design, default parameters, and
in places source code from three prior open-source projects. None of the
three is affiliated with or endorses OpenFold3.

The per-project sections below list exactly which files derive from what.
Files not listed in any section — `types.py`, `config.py`,
`batch_features.py`, and their tests — are original OpenFold3 code and
derive from none of these projects. `featurization.py` is mixed: its
extraction rules are Boltz-derived (listed below), its
`maybe_create_steering_features` entry point is not.

## Boltz

The flat-bottom potential formulation, the default restraint weights and
buffers (`defaults.py`), and the RDKit distance-geometry extraction rules
(`featurization.py`) are adapted from Boltz:

- `openfold3/steering/defaults.py`
- `openfold3/steering/potentials.py`
- `openfold3/steering/featurization.py`
- `openfold3/steering/engine.py`

<https://github.com/jwohlwend/boltz> (`src/boltz/model/potentials/potentials.py`,
`src/boltz/model/modules/diffusion.py`).

MIT License

Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Protenix

The separation of steering into a standalone, config-driven module with a
potential class registry (`CLASS_REGISTRY` / `register` in `potentials.py`)
follows the structure of `protenix/tfg/`. The chemistry criteria for terms not
yet implemented here (sp2/sp3 planarity, conjugated-torsion planarity) are
expected to follow the revised validity criteria from the Protenix-v2
technical report when those terms land.

<https://github.com/bytedance/Protenix>

Copyright (c) ByteDance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0. The full license text is at
<https://github.com/aqlaboratory/openfold-3/blob/main/LICENSE> (OpenFold3 is
itself Apache 2.0, so redistribution here needs no separate license copy per
License §4; this section preserves the attribution notice per §4(d)).

## foldsteer

The registry-driven engine (`engine.py`, `ChemicalSteering.on_denoised`), the
schedule types (`schedules.py`: `Constant`, `ExponentialInterpolation`,
`PiecewiseStepFunction`), and the broadcast-regression test sweep
(`tests/test_potentials.py`) are adapted from foldsteer, itself an independent
reimplementation deriving from Boltz and Protenix (see foldsteer's own
`NOTICE`).

- `openfold3/steering/engine.py`
- `openfold3/steering/schedules.py`
- `openfold3/steering/tests/test_potentials.py`

<https://github.com/etowahadams/foldsteer>

Copyright 2026 the foldsteer authors

Licensed under the Apache License, Version 2.0. The full license text is at
<https://github.com/aqlaboratory/openfold-3/blob/main/LICENSE> (see note
above); this section preserves the attribution notice per §4(d).

## RDKit

Constraint extraction uses RDKit (BSD 3-Clause), <https://www.rdkit.org/>.
