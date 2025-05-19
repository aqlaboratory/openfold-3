import logging
from typing import Any


def _ensure_list(value: Any) -> Any:
    if not isinstance(value, list):
        logging.info(f"Single value: {value} will be converted to a list")
        return [value]
    else:
        return value
