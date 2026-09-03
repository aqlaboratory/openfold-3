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

"""Regression tests for the post-run exit deadlock in ``run_openfold``.

Observed on a SLURM H100 node (8 ranks, ``run_openfold predict``): every
prediction was written and ``summary.txt`` said COMPLETE, then all 8 ranks sat
at 0% GPU utilisation holding their GPU memory until the scheduler killed the
job. ``PYTHONFAULTHANDLER=1`` plus ``kill -ABRT`` on a rank gave::

    File "multiprocessing/util.py",           line 428 in _exit_function
    File "multiprocessing/process.py",        line 156 in join
    File "multiprocessing/popen_fork.py",     line  44 in wait
    File "multiprocessing/popen_forkserver.py", line 65 in poll
    File "multiprocessing/connection.py",     line 1190 in wait
    File "selectors.py",                      line 398 in select

The C stack placed this under ``Py_Exit``: it is interpreter shutdown, and
``multiprocessing``'s atexit handler is blocked in an *untimed*
``Process.join()`` on a DataLoader worker.

The mechanism is specific to the ``forkserver`` start method. Workers started
that way are children of the forkserver process, not of the rank, so a rank
cannot ``waitpid()`` them itself -- it waits on a sentinel pipe for the
forkserver to report the exit status. On the hung node the forkserver was idle
in ``ep_poll`` and all ten workers were zombies (``Z``), i.e. already dead but
never reaped, so that status was never written and ``join()`` waited forever.

CPU-only by construction: no CUDA, no distributed backend.
"""

import subprocess
import sys
import textwrap

import pytest

from openfold3.core.data.framework.data_module import DataModuleConfig

# Generous relative to the ~10s a healthy run takes; the failure mode is an
# unbounded hang, so any finite bound separates the two.
EXIT_TIMEOUT_S = 120


def test_default_multiprocessing_context_does_not_reparent_workers():
    """The default start method must leave workers as children of the rank.

    ``forkserver`` reparents workers onto the forkserver process, which is what
    makes the untimed ``Process.join()`` in ``multiprocessing.util._exit_function``
    able to hang forever when the forkserver stops reaping.

    This is the red test: ``safe_multiprocessing_context`` currently hardcodes
    ``"forkserver"`` on Linux (data_module.py:199-200).
    """
    resolved = DataModuleConfig.safe_multiprocessing_context(
        "openfold-default", num_workers=10
    )

    assert resolved != "forkserver", (
        "Default multiprocessing context resolved to 'forkserver'. Workers are "
        "then children of the forkserver, not of the rank, so a rank must wait "
        "on the forkserver to reap them and report the exit status. When that "
        "does not happen the workers become zombies and the untimed "
        "Process.join() in multiprocessing.util._exit_function deadlocks at "
        "interpreter shutdown."
    )


def test_num_workers_zero_needs_no_context():
    """Guard the one case that is deadlock-free regardless of start method."""
    assert (
        DataModuleConfig.safe_multiprocessing_context("openfold-default", num_workers=0)
        is None
    )


@pytest.mark.slow
@pytest.mark.parametrize("context", ["openfold-default", "spawn"])
def test_process_exits_after_dataloader_outlives_the_run(tmp_path, context):
    """A process holding a live DataLoader at exit must still terminate.

    Keeping the DataLoader referenced at interpreter shutdown is what the real
    entry point does (Lightning holds the datamodule), and it is what routes
    worker teardown through ``multiprocessing.util._exit_function`` rather than
    through PyTorch's own ``_shutdown_workers``. That atexit join is the frame
    the production stack died in.

    Honest caveat: this reproduces the *shape* of the failure but has not been
    observed to hang standalone -- it passes today even with ``forkserver``. The
    production deadlock needs some ingredient not isolated here (it was seen
    only with 8 ranks under SLURM). Treat this as a regression guard on the exit
    contract, not as proof of the bug; the assertion above is the real red test.
    """
    script = tmp_path / "exit_probe.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import torch
            from torch.utils.data import DataLoader, Dataset
            from openfold3.core.data.framework.data_module import DataModuleConfig

            KEEP = {{}}

            class DS(Dataset):
                def __len__(self):
                    return 128

                def __getitem__(self, i):
                    return torch.zeros(4)

            def main():
                ctx = DataModuleConfig.safe_multiprocessing_context(
                    {context!r}, num_workers=4
                )
                dl = DataLoader(
                    DS(), batch_size=4, num_workers=4, multiprocessing_context=ctx
                )
                # Outlive main(): force teardown through the atexit handler.
                KEEP["dl"] = dl
                for _ in dl:
                    pass
                print("work done", flush=True)

            if __name__ == "__main__":
                main()
            """
        )
    )

    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=EXIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"Process with a live DataLoader (context={context!r}) did not exit "
            f"within {EXIT_TIMEOUT_S}s. This is the production hang: work "
            "completes, then multiprocessing's atexit Process.join() blocks "
            "forever waiting on a worker that was never reaped."
        )

    assert "work done" in completed.stdout
    assert completed.returncode == 0, (
        f"exit code {completed.returncode}\nstderr:\n{completed.stderr[-2000:]}"
    )
