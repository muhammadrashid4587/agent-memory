"""Pure-Python TF-IDF vectorizer — no numpy or sklearn required."""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from typing import Dict, List, Optional, Tuple


def _tokenize(text: str) -> List[str]:
    """Lowercase and split text into alpha-numeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFVectorizer:
    """Incremental TF-IDF vectorizer that is fully thread-safe.

    Maintains a vocabulary built from all documents seen so far and can
    produce sparse-style dense vectors suitable for cosine-similarity
    search.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # token -> index in the vocabulary
        self._vocab: Dict[str, int] = {}
        # document frequency: token -> number of docs containing it
        self._df: Dict[str, int] = Counter()
        # total number of documents added
        self._n_docs: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        with self._lock:
            return len(self._vocab)

    def fit_transform(self, text: str) -> List[float]:
        """Add *text* to the corpus and return its TF-IDF vector.

        The vector length equals the current vocabulary size at the time of
        the call.  Previously-returned vectors may be shorter; callers must
        zero-pad when comparing.
        """
        tokens = _tokenize(text)
        if not tokens:
            with self._lock:
                return [0.0] * len(self._vocab) if self._vocab else [0.0]

        unique_tokens = set(tokens)

        with self._lock:
            # Update corpus statistics
            self._n_docs += 1
            for t in unique_tokens:
                self._df[t] = self._df.get(t, 0) + 1
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)

            return self._tfidf_vector(tokens)

    def transform(self, text: str) -> List[float]:
        """Return the TF-IDF vector for *text* without updating the corpus."""
        tokens = _tokenize(text)
        with self._lock:
            if not tokens or not self._vocab:
                return [0.0] * len(self._vocab) if self._vocab else [0.0]
            return self._tfidf_vector(tokens)

    # ------------------------------------------------------------------
    # Internal helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _tfidf_vector(self, tokens: List[str]) -> List[float]:
        tf = Counter(tokens)
        total = len(tokens)
        vec = [0.0] * len(self._vocab)
        n = max(self._n_docs, 1)
        for token, count in tf.items():
            idx = self._vocab.get(token)
            if idx is None:
                continue
            tf_val = count / total
            idf_val = math.log((1 + n) / (1 + self._df.get(token, 0))) + 1.0
            vec[idx] = tf_val * idf_val
        return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors, zero-padding the shorter one."""
    length = max(len(a), len(b))
    if length == 0:
        return 0.0
    # Pad to same length
    if len(a) < length:
        a = a + [0.0] * (length - len(a))
    if len(b) < length:
        b = b + [0.0] * (length - len(b))

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
