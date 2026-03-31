import importlib
import os
from pathlib import Path

PLACEHOLDER_PATH = "placeholder"


def prep_deepspeed():
    # deepspeed requires the envvar set, but doesn't care about value
    # if already set (i.e. local cutlass setup), retain existing value
    os.environ["CUTLASS_PATH"] = os.environ.get("CUTLASS_PATH", PLACEHOLDER_PATH)


def prep_cutlass():
    cutlass_lib_is_installed = importlib.util.find_spec("cutlass_library") is not None
    cutlass_path = Path(os.environ.get("CUTLASS_PATH", PLACEHOLDER_PATH))

    # TODO: This check is for backward compatibility with the old local cutlass setup.
    #  Remove this and use pip installation only in the future.
    if not cutlass_lib_is_installed:
        if not cutlass_path.exists():
            raise OSError(
                "CUTLASS_PATH environment variable is not set to a valid path, "
                "and cutlass_library is not installed. Please install nvidia-cutlass"
                "via pip or set CUTLASS_PATH to the root of a local cutlass clone."
            )

        return

    # apparently need to set the headers for cutlass
    import cutlass_library

    headers_dir = Path(cutlass_library.__file__).parent / "source/include"
    cpath = os.environ.get("CPATH", "")
    # TODO: technically, this test should be a little fancier
    if str(headers_dir.resolve()) not in cpath:
        if cpath:
            cpath += ":"

        os.environ["CPATH"] = cpath + str(headers_dir.resolve())
