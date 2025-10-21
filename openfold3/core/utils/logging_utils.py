# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for managing logs from multiple processes and adding context.

This module helps solve two common problems:
1. Collecting log messages from worker processes (created using `multiprocessing`)
   into a single stream handled by the main process (typically writing to a file).
   This is done using a queue and the `setup_worker_logging` function.
2. Adding temporary, contextual information (like a specific PDB ID or step name)
   to log messages generated during a particular task, without needing to pass
   that information through every function call. This uses `log_context`,
   `set_log_context`, and `ContextInjectingFilter`.
"""

import contextvars
import logging
import multiprocessing.queues
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

# Acts as temporary storage for context relevant to the current task.
# Should be set to a dictionary like {'pdb_id': '1abc', 'step': 'parse'} using
# the `set_log_context` context manager. Read by the `ContextInjectingFilter`.
log_context = contextvars.ContextVar("log_context", default=None)


@contextmanager
def set_log_context(context_items: dict[str, Any] | None):
    """
    Context manager to temporarily tag log messages with context info.

    Use this with a 'with' statement to set context items (like specific IDs)
    that the `ContextInjectingFilter` will add to log records created within
    the 'with' block.

    Example:
        with set_log_context({'pdb_id': '1abc'}):
            logger.info("Processing started.") # Log record gets 'pdb_id' attribute

    Args:
        context_items:
            A dictionary containing the context key-value pairs (e.g.,
            {'pdb_id': '1xyz'}) or None to clear context.
    """
    token = log_context.set(context_items)
    try:
        # Code inside the 'with' block runs here
        yield
    finally:
        # Ensures context is always reset, even if errors occur
        log_context.reset(token)


def setup_worker_logging(
    log_queue: multiprocessing.queues.Queue,
    logger_name: str = "openfold3",
    log_level: int | None = None,
    expected_context_keys: Iterable | None = None,
) -> None:
    """
    Prepares logging for a worker process to send logs via a queue.

    Configures the specified logger within a worker process. It removes any
    existing handlers and sets up logging as follows:
    - Adds a QueueHandler: Sends all log records to the main process via `log_queue`.
    - Adds a ContextInjectingFilter: Intercepts records sent to the queue. It
      checks `log_context` (set via `set_log_context`) and adds the context
      (e.g., `pdb_id`) to the record. Also adds default values for any keys
      listed in `expected_context_keys` if they weren't found in the current
      context, preventing errors in the main process's formatter.
    - Disables Propagation: Stops logs from potentially being handled twice (e.g.,
      by a default console logger in the worker).
    - Sets Level: Configures the minimum severity level for the logger.

    Args:
        log_queue:
            The `multiprocessing.Queue` instance used to send logs back to the
            main process's listener.
        logger_name:
            The name of the logger to configure (should match the namespace
            used by logging calls in the worker code, e.g., "openfold3").
        log_level:
            The integer logging level (e.g., `logging.INFO`, `logging.DEBUG`)
            for this logger. If None, the level is not explicitly set here.
        expected_context_keys:
            An iterable (like a set or list) of context keys (strings like
            "pdb_id") that the log formatter in the main process expects. This
            tells the `ContextInjectingFilter` which keys need a default value
            added if they aren't currently set in `log_context`.
    """
    logger = logging.getLogger(logger_name)

    # Remove any existing handlers
    logger.handlers.clear()

    # Send log records over the queue to the listener
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # Inject context information before queuing
    context_filter = ContextInjectingFilter(
        expected_keys=expected_context_keys,
    )
    queue_handler.addFilter(context_filter)

    # Stop logs processed here from also going to the root logger's handlers
    logger.propagate = False

    if log_level is not None:
        logger.setLevel(log_level)


class ContextInjectingFilter(logging.Filter):
    """
    A logging filter that adds context information to log records.

    It reads a dictionary from the `log_context` ContextVar (if set via
    `set_log_context`) and adds each key-value pair as an attribute
    directly onto the `logging.LogRecord` object (e.g., adds `record.pdb_id`).

    It also ensures that a default value is set for any keys specified in
    `expected_keys` during initialization, preventing errors if the main
    log formatter uses a key that isn't currently set in the context.
    """

    def __init__(
        self,
        name: str = "",
        expected_keys: Iterable | None = None,
        default_value: Any = "-",
    ):
        """
        Initialize the filter.

        Args:
            name:
                Standard logging.Filter name (usually "").
            expected_keys:
                An iterable of keys (str) that the main log formatter might use
                (e.g., {'pdb_id', 'step'}). If a key from this set is not found
                in the current `log_context` dictionary, this filter will add
                it to the log record with the `default_value`.
            default_value:
                The value to use for expected keys that are missing from the
                current context (default is '-').
        """
        super().__init__(name=name)
        # Store expected keys as a set for efficient lookup
        self.expected_keys = set(expected_keys) if expected_keys is not None else set()
        self.default_value = default_value

    def filter(self, record: logging.LogRecord) -> True:
        """
        Adds context attributes and defaults to the log record.

        Returns:
            True: Always allows the record to be processed further.
        """
        context_dict = log_context.get()

        # Add attributes to the record from the current context dict
        if context_dict:
            for key, value in context_dict.items():
                setattr(record, key, value)

        # Ensure all keys expected by the formatter have at least a default value to
        # avoid an AttributeError (otherwise context needs to always be set)
        for key in self.expected_keys:
            if not hasattr(record, key):
                setattr(record, key, self.default_value)

        # Allow the record to pass through the filter
        return True
