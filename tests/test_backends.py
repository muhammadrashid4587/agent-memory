"""Tests for backend implementations (MemoryBackend and SQLiteBackend)."""

import os
import tempfile
import time

import pytest

from agent_memory.backends.memory_backend import MemoryBackend
from agent_memory.backends.sqlite_backend import SQLiteBackend
from agent_memory.memory import AgentMemory


# ── Parametrized backend tests ────────────────────────────────────────


def _make_memory_backend():
    return MemoryBackend()


def _make_sqlite_backend():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    backend = SQLiteBackend(path=path)
    backend._test_path = path  # stash for cleanup
    return backend


@pytest.fixture(params=["memory", "sqlite"])
def backend(request):
    if request.param == "memory":
        b = _make_memory_backend()
    else:
        b = _make_sqlite_backend()
    yield b
    b.close()
    if hasattr(b, "_test_path"):
        try:
            os.unlink(b._test_path)
        except OSError:
            pass


class TestBackendMemories:
    def test_save_and_load(self, backend):
        mem = {
            "id": "m1",
            "text": "hello",
            "metadata": {"k": "v"},
            "embedding": [0.1, 0.2],
            "created_at": time.time(),
        }
        backend.save_memory(mem)
        loaded = backend.load_memories()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "m1"
        assert loaded[0]["text"] == "hello"

    def test_delete(self, backend):
        mem = {
            "id": "m1",
            "text": "temp",
            "metadata": {},
            "embedding": [],
            "created_at": time.time(),
        }
        backend.save_memory(mem)
        assert backend.delete_memory("m1") is True
        assert backend.load_memories() == []

    def test_delete_nonexistent(self, backend):
        assert backend.delete_memory("nope") is False


class TestBackendMessages:
    def test_save_and_load(self, backend):
        msg = {"role": "user", "content": "hi", "timestamp": time.time(), "metadata": {}}
        backend.save_message(msg)
        loaded = backend.load_messages()
        assert len(loaded) == 1
        assert loaded[0]["content"] == "hi"

    def test_last_n(self, backend):
        for i in range(5):
            backend.save_message(
                {"role": "user", "content": str(i), "timestamp": time.time(), "metadata": {}}
            )
        loaded = backend.load_messages(last_n=2)
        assert len(loaded) == 2
        assert loaded[0]["content"] == "3"
        assert loaded[1]["content"] == "4"

    def test_clear(self, backend):
        backend.save_message(
            {"role": "user", "content": "x", "timestamp": time.time(), "metadata": {}}
        )
        backend.clear_messages()
        assert backend.load_messages() == []


class TestBackendKV:
    def test_save_and_load(self, backend):
        entry = {"key": "k1", "value": 42, "ttl": None, "created_at": time.time()}
        backend.save_kv("k1", entry)
        loaded = backend.load_kv("k1")
        assert loaded is not None
        assert loaded["value"] == 42

    def test_upsert(self, backend):
        entry1 = {"key": "k1", "value": 1, "ttl": None, "created_at": time.time()}
        entry2 = {"key": "k1", "value": 2, "ttl": None, "created_at": time.time()}
        backend.save_kv("k1", entry1)
        backend.save_kv("k1", entry2)
        loaded = backend.load_kv("k1")
        assert loaded["value"] == 2

    def test_delete(self, backend):
        entry = {"key": "k1", "value": "x", "ttl": None, "created_at": time.time()}
        backend.save_kv("k1", entry)
        assert backend.delete_kv("k1") is True
        assert backend.load_kv("k1") is None

    def test_list_keys(self, backend):
        for k in ["a", "b", "c"]:
            backend.save_kv(k, {"key": k, "value": k, "ttl": None, "created_at": time.time()})
        assert sorted(backend.list_kv_keys()) == ["a", "b", "c"]


# ── Integration: AgentMemory with SQLite persistence ──────────────────


class TestAgentMemoryIntegration:
    def test_sqlite_persistence(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        # Session 1: store data
        with AgentMemory(backend="sqlite", path=db_path) as mem:
            mem.store("The user likes cats", metadata={"type": "preference"})
            mem.add_message("user", "Hello")
            mem.set("theme", "dark")

        # Session 2: data should survive
        with AgentMemory(backend="sqlite", path=db_path) as mem:
            assert mem.memory_count() == 1
            results = mem.search("cats")
            assert len(results) >= 1
            assert "cats" in results[0].memory.text

            history = mem.get_history()
            assert len(history) == 1
            assert history[0].content == "Hello"

            assert mem.get("theme") == "dark"

    def test_memory_backend_is_ephemeral(self):
        mem1 = AgentMemory(backend="memory")
        mem1.store("temporary")
        mem1.close()

        mem2 = AgentMemory(backend="memory")
        assert mem2.memory_count() == 0
        mem2.close()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            AgentMemory(backend="redis")

    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx.db")
        with AgentMemory(backend="sqlite", path=db_path) as mem:
            mem.store("test")
            assert mem.memory_count() == 1

    def test_full_workflow(self):
        with AgentMemory(backend="memory") as mem:
            # Store memories
            mem.store("User prefers dark mode", metadata={"type": "preference"})
            mem.store("User's name is Alice", metadata={"type": "identity"})

            # Search
            results = mem.search("what does the user like?", top_k=3)
            assert len(results) >= 1

            # Conversation
            mem.add_message("user", "Hello!")
            mem.add_message("assistant", "Hi there!")
            history = mem.get_history(last_n=10)
            assert len(history) == 2

            # KV
            mem.set("last_tool", "web_search", ttl=3600)
            assert mem.get("last_tool") == "web_search"

            # Delete
            assert mem.delete_memory(results[0].memory.id) is True
            assert mem.memory_count() == 1
