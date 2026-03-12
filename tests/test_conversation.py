"""Tests for ConversationHistory."""

import pytest

from agent_memory.backends.memory_backend import MemoryBackend
from agent_memory.conversation import ConversationHistory, _estimate_tokens


class TestEstimateTokens:
    def test_short_text(self):
        assert _estimate_tokens("hi") == 1  # min 1

    def test_longer_text(self):
        text = "a" * 100
        assert _estimate_tokens(text) == 25

    def test_empty(self):
        assert _estimate_tokens("") == 1  # min 1


class TestConversationHistory:
    def _make_history(self, max_tokens=None) -> ConversationHistory:
        return ConversationHistory(MemoryBackend(), max_tokens=max_tokens)

    def test_add_and_retrieve(self):
        ch = self._make_history()
        ch.add_message("user", "Hello!")
        ch.add_message("assistant", "Hi there!")
        history = ch.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_last_n(self):
        ch = self._make_history()
        for i in range(10):
            ch.add_message("user", f"msg {i}")
        history = ch.get_history(last_n=3)
        assert len(history) == 3
        assert history[0].content == "msg 7"

    def test_max_tokens_truncation(self):
        ch = self._make_history(max_tokens=10)
        # Each 4-char message costs ~1 token
        # A longer message costs more
        ch.add_message("user", "A" * 40)  # ~10 tokens
        ch.add_message("user", "B" * 40)  # ~10 tokens
        ch.add_message("user", "C" * 20)  # ~5 tokens
        history = ch.get_history()
        # Budget is 10 tokens; "C" (5 tok) + "B" (10 tok) = 15 > 10
        # So only the last message should fit
        assert len(history) <= 2

    def test_max_tokens_override(self):
        ch = self._make_history(max_tokens=1000)
        ch.add_message("user", "A" * 40)  # ~10 tokens
        ch.add_message("user", "B" * 40)  # ~10 tokens
        # Override with a tight budget
        history = ch.get_history(max_tokens=10)
        assert len(history) == 1

    def test_clear(self):
        ch = self._make_history()
        ch.add_message("user", "hello")
        ch.clear()
        assert ch.get_history() == []

    def test_message_metadata(self):
        ch = self._make_history()
        msg = ch.add_message("user", "test", metadata={"source": "api"})
        assert msg.metadata["source"] == "api"

    def test_message_roles(self):
        ch = self._make_history()
        ch.add_message("system", "You are helpful.")
        ch.add_message("user", "Hi")
        ch.add_message("assistant", "Hello!")
        ch.add_message("tool", '{"result": 42}')
        history = ch.get_history()
        roles = [m.role for m in history]
        assert roles == ["system", "user", "assistant", "tool"]

    def test_empty_history(self):
        ch = self._make_history()
        assert ch.get_history() == []
        assert ch.get_history(last_n=5) == []
