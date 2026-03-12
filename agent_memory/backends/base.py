"""Abstract backend interface."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class Backend(abc.ABC):
    """Abstract storage backend.

    Every backend must support three logical tables:
      - **memories** (vector store entries)
      - **messages** (conversation messages)
      - **kv** (key-value pairs)
    """

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def save_memory(self, memory_dict: Dict[str, Any]) -> None:
        """Persist a single memory entry."""

    @abc.abstractmethod
    def load_memories(self) -> List[Dict[str, Any]]:
        """Return all stored memories."""

    @abc.abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by id. Return True if found."""

    # ------------------------------------------------------------------
    # Messages (conversation)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def save_message(self, message_dict: Dict[str, Any]) -> None:
        """Persist a conversation message."""

    @abc.abstractmethod
    def load_messages(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return stored messages, optionally limited to the last *n*."""

    @abc.abstractmethod
    def clear_messages(self) -> None:
        """Delete all conversation messages."""

    # ------------------------------------------------------------------
    # Key-Value
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def save_kv(self, key: str, entry_dict: Dict[str, Any]) -> None:
        """Persist a key-value entry (upsert)."""

    @abc.abstractmethod
    def load_kv(self, key: str) -> Optional[Dict[str, Any]]:
        """Return a single KV entry dict or None."""

    @abc.abstractmethod
    def delete_kv(self, key: str) -> bool:
        """Delete a KV entry. Return True if found."""

    @abc.abstractmethod
    def list_kv_keys(self) -> List[str]:
        """Return all stored KV keys."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources. Default is a no-op."""
