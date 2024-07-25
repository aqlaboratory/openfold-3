# TODO add license

MODEL_REGISTRY = {}


def register_model(cls):
    """Register a specific OpenFoldModelWrapper class in the MODEL_REGISTRY.

    Args:
        cls (Type[OpenFoldModelWrapper]): The class to register.

    Returns:
        Type[OpenFoldModelWrapper]: The registered class.
    """
    MODEL_REGISTRY[cls.__name__] = cls
    # cls._registered = True  # QUESTION do we want to enforce class registration with
    # this decorator? Part A.
    return cls
