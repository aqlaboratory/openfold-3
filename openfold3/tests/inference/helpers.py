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

"""Shared helpers for the end-to-end inference tests in this package.

Everything here drives a real ``InferenceExperimentRunner``, so it requires an
accelerator and downloaded model weights; the test modules gate on that with
``skip_unless_accelerator_available``.
"""

import logging
import os
import statistics
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from unittest.mock import patch

import pytest

import openfold3
from openfold3.core.config import config_utils
from openfold3.core.metrics.alignment import Structure
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import (
    InferenceExperimentConfig,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)

logger = logging.getLogger(__name__)

#: The metrics object a per-sample measurement returns — whatever ``core.metrics``
#: hands back (``SuperpositionMetrics``, ``LigandPoseMetrics``, ...).
T = TypeVar("T")

#: Committed experimental structures, used both as template inputs and RMSD references.
MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"

#: Seed the outputs are written under. Comes from ``ExperimentSettings.seeds`` (default
#: ``[42]``), which the runner yaml in :func:`run_inference` does not override — *not*
#: from ``InferenceQuerySet.seeds``, which the runner ignores.
SEED = 42

#: Diffusion samples drawn for a case whose prediction is scored against experiment.
#:
#: Accuracy is asserted on the *mean* over this many samples, never on one of them.
#: A single sample's RMSD is not reproducible across backends: diffusion noise is drawn
#: on-device (``diffusion_module.sample_diffusion``, ``centre_random_augmentation``), so
#: each accelerator has its own RNG stream and the same seed replays a different draw.
#: On ubiquitin the per-sample CA-RMSD spans 0.79-1.72 Å, an order of magnitude wider
#: than the accuracy differences these tests exist to detect.
#:
#: 8 rather than more because the samples share one trunk pass: raising N shrinks only
#: the within-trunk spread, and the run-to-run and cross-device terms — which it cannot
#: touch — are already the larger ones. Costs ~1.4 s per extra sample against a ~36 s
#: trunk, so the whole increase is a rounding error on an already-slow test.
SCORED_DIFFUSION_SAMPLES = 8


@dataclass(frozen=True)
class Mode:
    """One inference feature combination.

    These are the two independent branches of the data module's ``prepare_data``, and
    each materially changes how close a prediction lands to the experimental structure:
    with neither, the model runs single-sequence and for most targets misses the native
    fold entirely. So this is the axis accuracy expectations are keyed on — a single
    RMSD ceiling per target would be meaningless across all four.

    Frozen (and therefore hashable) so it can key a per-mode threshold table.
    """

    use_msa_server: bool
    use_templates: bool

    @property
    def id(self) -> str:
        """Short label used for the pytest parameter id."""
        msa = "msa" if self.use_msa_server else "no_msa"
        templates = "templates" if self.use_templates else "no_templates"
        return f"{msa}-{templates}"


#: Every combination — the axis inference cases are parametrized over.
MODES = tuple(
    Mode(use_msa_server=use_msa_server, use_templates=use_templates)
    for use_msa_server in (False, True)
    for use_templates in (False, True)
)


def query_set_from_chains(query_name: str, *chains: Mapping) -> InferenceQuerySet:
    """Build a single-query :class:`InferenceQuerySet` from raw chain dicts."""
    return InferenceQuerySet.model_validate(
        {"queries": {query_name: {"chains": list(chains)}}}
    )


def prediction_dir(output_dir: Path, query_name: str, *, seed: int = SEED) -> Path:
    """Directory the runner writes one query's predictions into.

    Mirrors ``InferenceExperimentRunner``: ``<output_dir>/<query>/seed_<seed>/``.
    """
    return output_dir / query_name / f"seed_{seed}"


def prediction_stem(query_name: str, sample: int, *, seed: int = SEED) -> str:
    """Filename prefix shared by one diffusion sample's output files."""
    return f"{query_name}_seed_{seed}_sample_{sample}"


def predicted_structure_cifs(
    output_dir: Path, query_name: str, *, seed: int = SEED
) -> list[Path]:
    """Predicted model cifs for one query, ordered by diffusion sample number.

    Sorted numerically rather than lexicographically so sample 10 does not land between
    1 and 2.
    """
    directory = prediction_dir(output_dir, query_name, seed=seed)
    return sorted(
        directory.glob(f"{query_name}_seed_{seed}_sample_*_model.cif"),
        key=lambda path: int(path.stem.rsplit("_sample_", 1)[1].split("_")[0]),
    )


@dataclass(frozen=True)
class SampleScores:
    """One metric measured on every diffusion sample of a single prediction.

    ``mean`` is what accuracy is asserted on; ``values`` and ``sd`` exist to be logged,
    since recalibrating a threshold needs the spread, not just the centre.
    """

    values: tuple[float, ...]

    @classmethod
    def of(
        cls, measurements: Iterable[T], select: Callable[[T], float]
    ) -> "SampleScores":
        """Pull one field out of per-sample measurements already taken."""
        return cls(tuple(select(measurement) for measurement in measurements))

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def sd(self) -> float:
        """Sample standard deviation, or 0.0 when there is only one sample."""
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def __str__(self) -> str:
        joined = ", ".join(f"{value:.2f}" for value in self.values)
        return (
            f"mean {self.mean:.2f} (sd {self.sd:.2f}, n={len(self.values)}) [{joined}]"
        )


def measure_samples(
    sample_cifs: Sequence[Path],
    measure: Callable[[Structure], T],
    *,
    expected_samples: int,
) -> list[T]:
    """Apply ``measure`` to every predicted sample, parsing each cif exactly once.

    ``measure`` takes the parsed prediction and returns whatever metrics object the
    caller wants — pull the individual fields out with :meth:`SampleScores.of` rather
    than calling this once per field, since parsing dominates the cost and the reference
    is the same for every sample (see ``core.metrics.alignment``).

    ``expected_samples`` is checked rather than trusted. Scoring whatever landed on disk
    would hide a partial write, and hide it in the worst direction: fewer samples widen
    the mean's standard error, so a run that lost samples is *less* likely to trip a
    ceiling than one that did not.
    """
    assert len(sample_cifs) == expected_samples, (
        f"Expected {expected_samples} predicted samples, found {len(sample_cifs)}: "
        f"{[cif.name for cif in sample_cifs]}"
    )
    return [measure(Structure.from_cif(cif)) for cif in sample_cifs]


inference_test_yaml_str = textwrap.dedent("""\
    model_update:
      presets:
        - predict
        - low_mem
    """)


def run_inference(
    query_set,
    output_dir: Path,
    *,
    use_msa_server: bool,
    use_templates: bool,
    num_diffusion_samples: int = 1,
    template_output_dir: Path | None = None,
) -> Path:
    """Run one inference job into ``output_dir`` and return it.

    Skips (``pytest.skip``) if no model checkpoint is available (escalated to a hard
    failure when ``OPENFOLD_SETUP_SCRIPT=1``). ``template_output_dir`` isolates the
    template cache per run (otherwise it lands in a persistent ``/tmp`` dir shared across
    runs and same-sequence queries).
    """
    runner_yaml = output_dir / "runner_config.yaml"
    yaml_str = inference_test_yaml_str
    if template_output_dir is not None:
        yaml_str += textwrap.dedent(f"""\
            template_preprocessor_settings:
              output_directory: {template_output_dir}
            """)
    runner_yaml.write_text(yaml_str)

    with patch("builtins.input", return_value="no"):
        experiment_config = InferenceExperimentConfig(
            **config_utils.load_yaml(runner_yaml)
        )
    runner = InferenceExperimentRunner(
        experiment_config,
        num_diffusion_samples=num_diffusion_samples,
        output_dir=output_dir,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
    )
    try:
        runner.setup()
    except ValueError as e:
        if "is not a valid file or directory" in str(e):
            if os.environ.get("OPENFOLD_SETUP_SCRIPT") == "1":
                raise AssertionError(
                    "No checkpoint files found after running setup script. "
                    "Please check that the download completed successfully."
                ) from None
            logger.warning(
                "No checkpoint files found, skipping. Use the setup script to "
                "download the weights."
            )
            pytest.skip("No checkpoint files available")
        raise

    runner.run(query_set)
    runner.cleanup()

    err_log_dir = output_dir / "logs"
    if err_log_dir.exists():
        raise RuntimeError(
            f"Found error logs in directory {err_log_dir}, "
            "check for errors in inference."
        )
    return output_dir
