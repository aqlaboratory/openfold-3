import importlib

if importlib.util.find_spec("deepspeed") is not None:
    import deepspeed

    # TODO: Resolve this
    # This is a hack to prevent deepspeed from doing the triton matmul autotuning
    # I'm not sure why it's doing this by default, but it's causing the tests to hang
    deepspeed.HAS_TRITON = False
