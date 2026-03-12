"""In-memory dict backend — fast, ephemeral, great for tests."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from agent_memory.backends.base import Backend


class MemoryBackend(Backend):
    """Store everything in plain Python dicts. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memories: Dict[str, Dict[str, Any]] = {}  # id -> dict
        self._messages: List[Dict[str, Any]] = []
        self._kv: Dict[str, Dict[str, Any]] = {}

    # -- Memories -------------------------------------------------------

    def save_memory(self, memory_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._memories[memory_dict["id"]] = memory_dict

    def load_memories(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._memories.values())

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            return self._memories.pop(memory_id, None) is not None

    # -- Messages -------------------------------------------------------

    def save_message(self, message_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._messages.append(message_dict)

    def load_messages(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if last_n is not None:
                return list(self._messages[-last_n:])
            return list(self._messages)

    def clear_messages(self) -> None:
        with self._lock:
            self._messages.clear()

    # -- Key-Value ------------------------------------------------------

    def save_kv(self, key: str, entry_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._kv[key] = entry_dict

    def load_kv(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._kv.get(key)

    def delete_kv(self, key: str) -> bool:
        with self._lock:
            return self._kv.pop(key, None) is not None

    def list_kv_keys(self) -> List[str]:
        with self._lock:
            return list(self._kv.keys())
