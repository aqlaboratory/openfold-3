"""Guards the conftest fixture that restores os.environ after every test.

The restore itself is ``unittest.mock.patch.dict``; what can silently break is the
fixture no longer wrapping every test (renamed, moved, or no longer autouse). Check
that directly, so the result does not depend on test order or xdist scheduling.
"""


def test_restore_environ_fixture_wraps_every_test(request):
    assert "_restore_environ" in request.fixturenames
