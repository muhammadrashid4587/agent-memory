"""KeyValueStore — typed get/set/delete with TTL support."""

from __future__ import annotations

import threading
import time
from typing import Any, List, Optional

from agent_memory.backends.base import Backend
from agent_memory.models import KVEntry


class KeyValueStore:
    """Simple key-value store with optional per-key TTL (time-to-live).

    Thread-safe: all operations are guarded by a lock.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store *value* under *key*.

        Parameters
        ----------
        key : str
            The key name.
        value : Any
            Must be JSON-serialisable.
        ttl : float or None
            Time-to-live in seconds.  ``None`` means no expiry.
        """
        entry = KVEntry(key=key, value=value, ttl=ttl)
        with self._lock:
            self._backend.save_kv(key, entry.model_dump())

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if missing / expired."""
        with self._lock:
            raw = self._backend.load_kv(key)
        if raw is None:
            return default
        entry = KVEntry(**raw)
        if entry.expired:
            self.delete(key)
            return default
        return entry.value

    def delete(self, key: str) -> bool:
        """Delete *key*. Return ``True`` if it existed."""
        with self._lock:
            return self._backend.delete_kv(key)

    def keys(self) -> List[str]:
        """Return all non-expired keys."""
        with self._lock:
            all_keys = self._backend.list_kv_keys()
        # Filter out expired ones
        alive: List[str] = []
        for k in all_keys:
            raw = self._backend.load_kv(k)
            if raw is None:
                continue
            entry = KVEntry(**raw)
            if not entry.expired:
                alive.append(k)
            else:
                self._backend.delete_kv(k)
        return alive

    def exists(self, key: str) -> bool:
        """Check if *key* exists and is not expired."""
        return self.get(key, _SENTINEL) is not _SENTINEL


# Sentinel object for existence checks
_SENTINEL = object()
