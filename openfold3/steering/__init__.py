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

"""Chemical steering: gradient guidance toward valid ligand chemistry.

See README.md for the architecture and provenance policy, and
THIRD_PARTY_NOTICES.md for full license texts.

This module deliberately exports only the sampling-loop side, which is
torch-only. `featurization` pulls in RDKit, biotite and the
data pipeline, so it is imported directly by the featurization path
rather than re-exported here -- otherwise every importer of this package,
including the model stack, would pay for them.
"""

from openfold3.steering.batch_features import (
    PreparedSteering,
    prepare_steering,
    steering_enabled,
)
from openfold3.steering.config import SteeringSettings, TermSettings
from openfold3.steering.engine import ChemicalSteering, Term
from openfold3.steering.types import RestraintSet, SteeringContext

__all__ = [
    "ChemicalSteering",
    "PreparedSteering",
    "RestraintSet",
    "SteeringContext",
    "SteeringSettings",
    "Term",
    "TermSettings",
    "prepare_steering",
    "steering_enabled",
]
