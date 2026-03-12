"""Tests for the VectorStore and TF-IDF embeddings."""

import pytest

from agent_memory.backends.memory_backend import MemoryBackend
from agent_memory.embeddings import TFIDFVectorizer, cosine_similarity
from agent_memory.vector import VectorStore


# ── TF-IDF Vectorizer ─────────────────────────────────────────────────


class TestTFIDFVectorizer:
    def test_basic_vectorization(self):
        v = TFIDFVectorizer()
        vec = v.fit_transform("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)

    def test_vocab_grows(self):
        v = TFIDFVectorizer()
        v.fit_transform("hello world")
        assert v.vocab_size == 2
        v.fit_transform("hello universe")
        assert v.vocab_size == 3  # "hello", "world", "universe"

    def test_transform_without_fit(self):
        v = TFIDFVectorizer()
        v.fit_transform("cat dog")
        vec = v.transform("cat fish")
        assert isinstance(vec, list)
        # "cat" should have a non-zero weight; "fish" is unknown → 0
        assert len(vec) == v.vocab_size

    def test_empty_text(self):
        v = TFIDFVectorizer()
        vec = v.fit_transform("")
        assert all(x == 0.0 for x in vec)

    def test_cosine_similarity_identical(self):
        a = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_different_lengths(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0]
        # Should zero-pad b to [1.0, 0.0, 0.0] → similarity = 1.0
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_cosine_similarity_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ── VectorStore ───────────────────────────────────────────────────────


class TestVectorStore:
    def _make_store(self) -> VectorStore:
        return VectorStore(MemoryBackend())

    def test_add_and_count(self):
        store = self._make_store()
        store.add("The sky is blue")
        store.add("Grass is green")
        assert store.count() == 2

    def test_search_returns_ranked(self):
        store = self._make_store()
        store.add("Python is a programming language")
        store.add("The weather is sunny today")
        store.add("I love writing Python code")

        results = store.search("Python programming", top_k=2)
        assert len(results) == 2
        # The most relevant result should mention Python
        assert "Python" in results[0].memory.text or "Python" in results[1].memory.text

    def test_search_top_k_limit(self):
        store = self._make_store()
        for i in range(10):
            store.add(f"Memory number {i}")
        results = store.search("memory", top_k=3)
        assert len(results) == 3

    def test_delete(self):
        store = self._make_store()
        mem = store.add("temporary memory")
        assert store.count() == 1
        assert store.delete(mem.id) is True
        assert store.count() == 0

    def test_delete_nonexistent(self):
        store = self._make_store()
        assert store.delete("nonexistent-id") is False

    def test_metadata(self):
        store = self._make_store()
        mem = store.add("dark mode preference", metadata={"type": "preference"})
        assert mem.metadata["type"] == "preference"

    def test_list_all(self):
        store = self._make_store()
        store.add("first")
        store.add("second")
        all_mems = store.list_all()
        assert len(all_mems) == 2

    def test_custom_id(self):
        store = self._make_store()
        mem = store.add("test", memory_id="custom-123")
        assert mem.id == "custom-123"

    def test_search_scores_in_range(self):
        store = self._make_store()
        store.add("hello world")
        store.add("goodbye world")
        results = store.search("hello")
        for r in results:
            assert 0.0 <= r.score <= 1.0
