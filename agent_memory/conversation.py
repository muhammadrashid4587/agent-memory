"""ConversationHistory — sliding window with token-aware truncation."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from agent_memory.backends.base import Backend
from agent_memory.models import Message


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token (GPT-style heuristic)."""
    return max(1, len(text) // 4)


class ConversationHistory:
    """Manages a conversation message list with optional token-window truncation.

    Parameters
    ----------
    backend : Backend
        The storage backend to persist messages.
    max_tokens : int or None
        If set, :meth:`get_history` will return only the most recent messages
        that fit within this token budget.
    """

    def __init__(self, backend: Backend, max_tokens: Optional[int] = None) -> None:
        self._backend = backend
        self._max_tokens = max_tokens
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Append a message to the conversation and persist it."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        with self._lock:
            self._backend.save_message(msg.model_dump())
        return msg

    def get_history(
        self,
        last_n: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Message]:
        """Retrieve conversation history.

        Parameters
        ----------
        last_n : int or None
            Return at most the last *n* messages.
        max_tokens : int or None
            Override the instance-level ``max_tokens``.  If provided, only the
            most recent messages that fit within this budget are returned.
        """
        raw = self._backend.load_messages(last_n=last_n)
        messages = [Message(**d) for d in raw]

        budget = max_tokens if max_tokens is not None else self._max_tokens
        if budget is not None:
            messages = self._truncate_to_budget(messages, budget)
        return messages

    def clear(self) -> None:
        """Delete all messages."""
        self._backend.clear_messages()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_to_budget(messages: List[Message], max_tokens: int) -> List[Message]:
        """Keep the most recent messages that fit within *max_tokens*."""
        result: List[Message] = []
        used = 0
        for msg in reversed(messages):
            cost = _estimate_tokens(msg.content)
            if used + cost > max_tokens:
                break
            result.append(msg)
            used += cost
        result.reverse()
        return result
