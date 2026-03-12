"""Tests for KeyValueStore."""

import time

import pytest

from agent_memory.backends.memory_backend import MemoryBackend
from agent_memory.kv import KeyValueStore


class TestKeyValueStore:
    def _make_kv(self) -> KeyValueStore:
        return KeyValueStore(MemoryBackend())

    def test_set_and_get(self):
        kv = self._make_kv()
        kv.set("name", "Alice")
        assert kv.get("name") == "Alice"

    def test_get_default(self):
        kv = self._make_kv()
        assert kv.get("missing") is None
        assert kv.get("missing", "fallback") == "fallback"

    def test_overwrite(self):
        kv = self._make_kv()
        kv.set("x", 1)
        kv.set("x", 2)
        assert kv.get("x") == 2

    def test_delete(self):
        kv = self._make_kv()
        kv.set("key", "val")
        assert kv.delete("key") is True
        assert kv.get("key") is None

    def test_delete_nonexistent(self):
        kv = self._make_kv()
        assert kv.delete("nope") is False

    def test_keys(self):
        kv = self._make_kv()
        kv.set("a", 1)
        kv.set("b", 2)
        kv.set("c", 3)
        assert sorted(kv.keys()) == ["a", "b", "c"]

    def test_exists(self):
        kv = self._make_kv()
        kv.set("x", "y")
        assert kv.exists("x") is True
        assert kv.exists("z") is False

    def test_ttl_not_expired(self):
        kv = self._make_kv()
        kv.set("temp", "data", ttl=3600)
        assert kv.get("temp") == "data"

    def test_ttl_expired(self):
        kv = self._make_kv()
        kv.set("temp", "data", ttl=0.01)
        time.sleep(0.05)
        assert kv.get("temp") is None

    def test_keys_filters_expired(self):
        kv = self._make_kv()
        kv.set("alive", "yes", ttl=3600)
        kv.set("dead", "no", ttl=0.01)
        time.sleep(0.05)
        assert kv.keys() == ["alive"]

    def test_complex_values(self):
        kv = self._make_kv()
        kv.set("config", {"theme": "dark", "font_size": 14})
        assert kv.get("config") == {"theme": "dark", "font_size": 14}

    def test_none_value(self):
        kv = self._make_kv()
        kv.set("empty", None)
        # Can't distinguish from missing using default=None, but exists works
        assert kv.exists("empty") is True

    def test_list_value(self):
        kv = self._make_kv()
        kv.set("items", [1, 2, 3])
        assert kv.get("items") == [1, 2, 3]
