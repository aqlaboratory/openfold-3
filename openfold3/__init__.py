__all__ = ["core", "projects"]

import importlib.util

if importlib.util.find_spec("deepspeed") is not None:
    import deepspeed

    # TODO: Resolve this later
    # This is a hack to prevent deepspeed from doing the triton matmul autotuning
    # This has weird effects with hanging if libaio is not installed and can
    # cause restart errors if run is preempted in the middle of autotuning
    deepspeed.HAS_TRITON = False
