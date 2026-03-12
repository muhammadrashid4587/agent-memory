"""AgentMemory — unified facade over vector store, conversation history, and KV state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_memory.backends.base import Backend
from agent_memory.backends.memory_backend import MemoryBackend
from agent_memory.backends.sqlite_backend import SQLiteBackend
from agent_memory.conversation import ConversationHistory
from agent_memory.kv import KeyValueStore
from agent_memory.models import Memory, Message, SearchResult
from agent_memory.vector import VectorStore


class AgentMemory:
    """High-level API that combines vector search, conversation history, and KV state.

    Parameters
    ----------
    backend : str
        ``"memory"`` for an ephemeral in-memory backend (default) or
        ``"sqlite"`` for a persistent SQLite backend.
    path : str
        Path to the SQLite database file (only used when ``backend="sqlite"``).
    max_tokens : int or None
        Optional token budget for conversation history windowing.
    """

    def __init__(
        self,
        backend: str = "memory",
        path: str = "agent_memory.db",
        max_tokens: Optional[int] = None,
    ) -> None:
        self._backend_instance: Backend
        if backend == "sqlite":
            self._backend_instance = SQLiteBackend(path=path)
        elif backend == "memory":
            self._backend_instance = MemoryBackend()
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'memory' or 'sqlite'.")

        self._vector = VectorStore(self._backend_instance)
        self._conversation = ConversationHistory(
            self._backend_instance, max_tokens=max_tokens
        )
        self._kv = KeyValueStore(self._backend_instance)

    # ------------------------------------------------------------------
    # Vector / semantic memory
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> Memory:
        """Store a new memory for later semantic retrieval."""
        return self._vector.add(text, metadata=metadata, memory_id=memory_id)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Semantically search stored memories."""
        return self._vector.search(query, top_k=top_k)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by id."""
        return self._vector.delete(memory_id)

    def memory_count(self) -> int:
        """Return the number of stored memories."""
        return self._vector.count()

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to the conversation history."""
        return self._conversation.add_message(role, content, metadata=metadata)

    def get_history(
        self,
        last_n: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Message]:
        """Retrieve conversation history with optional windowing."""
        return self._conversation.get_history(last_n=last_n, max_tokens=max_tokens)

    def clear_history(self) -> None:
        """Clear all conversation messages."""
        self._conversation.clear()

    # ------------------------------------------------------------------
    # Key-value state
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a key-value pair with optional TTL (seconds)."""
        self._kv.set(key, value, ttl=ttl)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key, returning *default* if missing or expired."""
        return self._kv.get(key, default=default)

    def delete_key(self, key: str) -> bool:
        """Delete a key-value entry."""
        return self._kv.delete(key)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release backend resources."""
        self._backend_instance.close()

    def __enter__(self) -> "AgentMemory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
