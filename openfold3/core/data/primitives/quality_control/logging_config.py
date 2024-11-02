import contextvars
import logging
import os
from pathlib import Path

TEMPLATE_PROCESS_LOGGER = contextvars.ContextVar(
    "TEMPLATE_PROCESS_LOGGER", default=None
)


def configure_template_logger(
    log_level: str, log_to_file: bool, log_to_console: bool, log_dir: Path
) -> logging.Logger:
    logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper())
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    if log_to_file:
        pid = os.getpid()
        file_handler = logging.FileHandler(log_dir / Path(f"process_{pid}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
