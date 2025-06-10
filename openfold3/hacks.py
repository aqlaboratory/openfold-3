import os
from pathlib import Path

# deepspeed requires the envvar set, but doesn't care about value
os.environ["CUTLASS_PATH"] = os.environ.get("CUTLASS_PATH", "placeholder")

# apparently need to set the headers for cutlass
import cutlass_library
headers_dir = Path(cutlass_library.__file__).parent / "source/include"
cpath = os.environ.get("CPATH", "")
if cpath:
    cpath += ":"

os.environ["CPATH"] = cpath + str(headers_dir.resolve())

