from __future__ import annotations

import json
import platform
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import biotite.setup_ccd
import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from torch.random import fork_rng

from openfold3.core.data.primitives.structure.component import BiotiteCCDWrapper
from openfold3.setup_openfold import setup_biotite_ccd

# ---------------------------------------------------------------------------
# Device fixture: parametrize tests to run on both CPU and CUDA
# ---------------------------------------------------------------------------

_DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]


@pytest.fixture(params=_DEVICES)
def device(request) -> str:
    """Yield 'cpu' or 'cuda'; CUDA tests are auto-skipped when no GPU."""
    return request.param


# ---------------------------------------------------------------------------
# CUDA determinism: ensure reproducible results on the same hardware
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cuda_deterministic(request):
    """Enable deterministic CUDA ops for tests that use the ``device`` fixture."""
    if "device" not in request.fixturenames:
        yield
        return

    dev = request.getfixturevalue("device")
    if dev != "cuda":
        yield
        return

    orig_deterministic = torch.backends.cudnn.deterministic
    orig_benchmark = torch.backends.cudnn.benchmark
    orig_det_algos = torch.are_deterministic_algorithms_enabled()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    yield

    torch.backends.cudnn.deterministic = orig_deterministic
    torch.backends.cudnn.benchmark = orig_benchmark
    torch.use_deterministic_algorithms(orig_det_algos)


# ---------------------------------------------------------------------------
# Snapshot environment metadata
# ---------------------------------------------------------------------------

_SNAPSHOT_ENV_FILE = "_snapshot_env.json"


@dataclass(frozen=True)
class SnapshotEnv:
    """Environment info relevant to snapshot reproducibility."""

    torch_version: str
    python_version: str
    cuda_version: str | None = None
    cudnn_version: str | None = None
    gpu_name: str | None = None

    @classmethod
    def current(cls) -> SnapshotEnv:
        cuda_kwargs = {}
        if torch.cuda.is_available():
            cuda_kwargs = dict(
                cuda_version=torch.version.cuda,
                cudnn_version=str(torch.backends.cudnn.version()),
                gpu_name=torch.cuda.get_device_name(0),
            )
        return cls(
            torch_version=torch.__version__,
            python_version=platform.python_version(),
            **cuda_kwargs,
        )

    @classmethod
    def from_json(cls, path: Path) -> SnapshotEnv:
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    def mismatches(self, other: SnapshotEnv) -> list[str]:
        result = []
        for field in ("torch_version", "cuda_version", "cudnn_version", "gpu_name"):
            stored = getattr(self, field)
            current = getattr(other, field)
            if stored is not None and current is not None and stored != current:
                result.append(f"  {field}: stored={stored}, current={current}")
        return result


def _check_snapshot_env(snapshot_dir: Path) -> None:
    """Warn if the current environment differs from the one that generated snapshots."""
    env_file = snapshot_dir / _SNAPSHOT_ENV_FILE
    if not env_file.exists():
        return

    stored = SnapshotEnv.from_json(env_file)
    mismatches = stored.mismatches(SnapshotEnv.current())

    if mismatches:
        warnings.warn(
            f"Snapshot environment mismatch in {snapshot_dir.name}/:\n"
            + "\n".join(mismatches)
            + "\nSnapshot tests may fail. Regenerate with: pytest --force-regen",
            stacklevel=2,
        )


def _write_snapshot_env(snapshot_dir: Path) -> None:
    """Write current environment metadata alongside snapshots."""
    SnapshotEnv.current().to_json(snapshot_dir / _SNAPSHOT_ENV_FILE)


def pytest_sessionfinish(session, exitstatus):
    """After ``--force-regen``, write environment metadata to snapshot dirs."""
    if not session.config.getoption("force_regen", default=False):
        return
    snapshots_root = Path(__file__).parent / "test_data" / "snapshots"
    if snapshots_root.exists():
        for subdir in snapshots_root.iterdir():
            if subdir.is_dir() and any(subdir.glob("*.npz")):
                _write_snapshot_env(subdir)


@pytest.fixture
def dummy_atom_array():
    # Create dummy atom array
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.4, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.4, 0.0, 0.0],
        ],
        dtype=float,
    )
    atom_array = AtomArray(len(coords))
    atom_array.coord = coords
    return atom_array


@pytest.fixture
def mse_ala_atom_array():
    """AtomArray with one MSE residue and one ALA residue for testing MSE->MET conversion."""
    # MSE has a selenium atom (SE), ALA is a simple residue for comparison
    # MSE atoms: N, CA, C, O, CB, CG, SE, CE (8 atoms)
    # ALA atoms: N, CA, C, O, CB (5 atoms)
    n_atoms = 13
    atom_array = AtomArray(n_atoms)

    atom_array.coord = np.zeros((n_atoms, 3))

    # MSE residue (res_id=1)
    atom_array.chain_id[:8] = "A"
    atom_array.res_id[:8] = 1
    atom_array.res_name[:8] = "MSE"
    atom_array.atom_name[:8] = ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"]
    atom_array.element[:8] = ["N", "C", "C", "O", "C", "C", "SE", "C"]
    atom_array.hetero[:8] = True

    # ALA residue (res_id=2)
    atom_array.chain_id[8:] = "A"
    atom_array.res_id[8:] = 2
    atom_array.res_name[8:] = "ALA"
    atom_array.atom_name[8:] = ["N", "CA", "C", "O", "CB"]
    atom_array.element[8:] = ["N", "C", "C", "O", "C"]
    atom_array.hetero[8:] = False

    return atom_array


def pytest_addoption(parser):
    parser.addoption(
        "--skip-ccd-update",
        action="store_true",
        default=False,
        help="Skip downloading/verifying the Biotite CCD file",
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_biotite_ccd(request):
    """Download CCD file before any tests run (once per test session)."""
    if request.config.getoption("--skip-ccd-update"):
        return
    setup_biotite_ccd(ccd_path=biotite.setup_ccd.OUTPUT_CCD, force_download=False)


@pytest.fixture(scope="session")
def biotite_ccd_wrapper():
    """Cache CCD wrapper fixture for tests that need it."""
    return BiotiteCCDWrapper()


@pytest.fixture(scope="module")
def original_datadir(request: pytest.FixtureRequest) -> Path:
    """Redirect pytest-regressions snapshot storage to test_data/snapshots/."""
    datadir = (
        Path(__file__).parent / "test_data" / "snapshots" / Path(request.path).stem
    )
    _check_snapshot_env(datadir)
    return datadir


@pytest.fixture()
def seeded_rng():
    """Isolate all RNG state (torch, numpy, python) for the duration of a test.

    Uses torch.random.fork_rng() to save/restore torch (+CUDA) state, and
    manually saves/restores numpy and python random state.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    with fork_rng():
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        yield
    random.setstate(py_state)
    np.random.set_state(np_state)
