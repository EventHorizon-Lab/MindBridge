"""Bounded immutable read results, valid only within one store's write generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from threading import Lock
from typing import Generic, TypeVar

_T = TypeVar("_T")


class ReadCache(Generic[_T]):
    """Reuse complete slates only; misses must reread every key before publishing."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._generation = -1
        self._reads = 0
        self._disabled = False
        self._values: dict[str, _T] = {}
        self._lock = Lock()

    def get(self, keys: Sequence[str], *, generation: int) -> tuple[_T, ...] | None:
        with self._lock:
            if self._generation != generation:
                self._values.clear()
                self._generation = generation
                self._reads = 0
                self._disabled = False
            self._reads += 1
            # Frequent commits make these caches churn without earning their fill cost.
            # Require more than a wave of concurrent requests before retaining a working set.
            if self._disabled or self._reads < 64:
                return None
            if keys and all(key in self._values for key in keys):
                return tuple(self._values[key] for key in keys)
            return None

    def put(self, values: Iterable[tuple[str, _T]], *, generation: int) -> None:
        with self._lock:
            # A reader from an older generation must not populate a newer reader's cache.
            if self._generation != generation or self._disabled or self._reads < 64:
                return
            for key, value in values:
                self._values[key] = value
                if len(self._values) > self._capacity:
                    # Whole-slate hits are unlikely when the working set does not fit. Stop
                    # paying fill/eviction costs until the next write generation.
                    self._values.clear()
                    self._disabled = True
                    return
