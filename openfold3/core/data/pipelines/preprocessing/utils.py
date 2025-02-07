"""Utility functions for data preprocessing pipelines."""

from collections.abc import Iterable
from multiprocessing import Manager
from multiprocessing.managers import SyncManager


class SharedSet:
    """A set-like object that can be shared between processes.

    Internally wraps a multiprocessing Manager dict.
    """

    def __init__(self, iterable: Iterable = None, manager: SyncManager | None = None):
        if manager is None:
            manager = Manager()

        self._set = manager.dict()

        if iterable is not None:
            self.update(iterable)

    def add(self, item):
        """Add an item to the shared set."""
        self._set[item] = None

    def remove(self, item):
        """Remove an item from the shared set."""
        if item in self._set:
            del self._set[item]
        else:
            raise KeyError(f"{item} not found in the set")

    def __contains__(self, item):
        """Check if an item is in the shared set."""
        return item in self._set

    def __len__(self):
        """Return the number of items in the shared set."""
        return len(self._set)

    def __iter__(self):
        """Return an iterator over the items in the shared set."""
        return iter(self._set.keys())

    def clear(self):
        """Clear all items from the shared set."""
        self._set.clear()

    def update(self, other: Iterable):
        """Update the set with the union of itself and an iterable."""

        self._set.update({k: None for k in other})
