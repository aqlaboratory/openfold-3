import os
from pathlib import Path


# Maybe it would be better to force people to set the ENV variable themselves?
def prep_deepspeed():
    # deepspeed requires the envvar set, but doesn't care about value
    os.environ["CUTLASS_PATH"] = os.environ.get(
        "CUTLASS_PATH", "DS_IGNORE_CUTLASS_DETECTION"
    )


def prep_cutlass():
    # apparently need to set the headers for cutlass
    try:
        import cutlass_library
        headers_dir = Path(cutlass_library.__file__).parent / "source/include"
        cpath = os.environ.get("CPATH", "")
        # TODO: technically, this test should be a little fancier
        if str(headers_dir.resolve()) not in cpath:
            if cpath:
                cpath += ":"

            os.environ["CPATH"] = cpath + str(headers_dir.resolve())
    except ImportError:
        # Assume we are in a conda setting that does not require the PATH modification
        # Unfortunately, if we are not, then the build will fail with a rather cryptic 
        # error message.
        pass

        


