"""Guards the conftest fixture that restores os.environ after every test.

The two tests below must run in order in the same process: the first leaks a
variable the way ``setup_openfold.run_setup`` leaks ``OPENFOLD_CACHE``, the second
checks it is gone. Under ``pytest -n`` they may land on different workers, in which
case the second passes without exercising anything.
"""

import os

_CANARY = "_OPENFOLD_TEST_ENVIRON_CANARY"


def test_leak_environ_variable():
    os.environ[_CANARY] = "leaked"


def test_environ_variable_did_not_leak():
    assert _CANARY not in os.environ
