"""VectorStore — cosine-similarity search over TF-IDF embeddings."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from agent_memory.backends.base import Backend
from agent_memory.embeddings import TFIDFVectorizer, cosine_similarity
from agent_memory.models import Memory, SearchResult


class VectorStore:
    """Stores text memories and retrieves them via semantic (TF-IDF) search.

    Thread-safe: all mutating operations are guarded by a lock.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._vectorizer = TFIDFVectorizer()
        self._memories: Dict[str, Memory] = {}
        self._lock = threading.Lock()
        self._load_from_backend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> Memory:
        """Store a new memory and return it."""
        embedding = self._vectorizer.fit_transform(text)
        mem = Memory(
            text=text,
            metadata=metadata or {},
            embedding=embedding,
            **({"id": memory_id} if memory_id else {}),
        )
        with self._lock:
            self._memories[mem.id] = mem
        self._backend.save_memory(mem.model_dump())
        return mem

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Return the *top_k* most similar memories for *query*."""
        query_vec = self._vectorizer.transform(query)
        scored: List[SearchResult] = []

        with self._lock:
            for mem in self._memories.values():
                score = cosine_similarity(query_vec, mem.embedding)
                scored.append(SearchResult(memory=mem, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by id."""
        with self._lock:
            removed = self._memories.pop(memory_id, None) is not None
        if removed:
            self._backend.delete_memory(memory_id)
        return removed

    def count(self) -> int:
        with self._lock:
            return len(self._memories)

    def list_all(self) -> List[Memory]:
        with self._lock:
            return list(self._memories.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_from_backend(self) -> None:
        """Hydrate in-memory state from the backend on startup."""
        for d in self._backend.load_memories():
            mem = Memory(**d)
            self._memories[mem.id] = mem
            # Re-fit the vectorizer so IDF counts are correct
            self._vectorizer.fit_transform(mem.text)
