import numpy as np


def encode_numpy_types(obj: object):
    """An encoding function for NumPy -> standard types.

    This is useful for JSON serialisation for example, which can't deal with NumPy
    types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def is_intlike_string(value: str) -> bool:
    """Check if a string represents an integer.

    Args:
        value:
            The string to check.

    Returns:
        Whether the string represents an integer.
    """
    try:
        int(value)
        return True
    except ValueError:
        return False
