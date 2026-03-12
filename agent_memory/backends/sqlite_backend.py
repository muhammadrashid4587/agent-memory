"""SQLite persistent backend with WAL mode for concurrent reads."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from agent_memory.backends.base import Backend


class SQLiteBackend(Backend):
    """Persistent backend using a single SQLite database file.

    Uses WAL journal mode for better concurrent-read performance.
    All public methods are thread-safe.
    """

    def __init__(self, path: str = "agent_memory.db") -> None:
        self._path = path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ttl REAL,
                created_at REAL NOT NULL
            );
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def save_memory(self, memory_dict: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO memories (id, text, metadata, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    memory_dict["id"],
                    memory_dict["text"],
                    json.dumps(memory_dict.get("metadata", {})),
                    json.dumps(memory_dict.get("embedding", [])),
                    memory_dict["created_at"],
                ),
            )
            conn.commit()

    def load_memories(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT * FROM memories ORDER BY created_at").fetchall()
            return [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": json.loads(r["metadata"]),
                    "embedding": json.loads(r["embedding"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def save_message(self, message_dict: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO messages (role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
                (
                    message_dict["role"],
                    message_dict["content"],
                    message_dict["timestamp"],
                    json.dumps(message_dict.get("metadata", {})),
                ),
            )
            conn.commit()

    def load_messages(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            if last_n is not None:
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM messages ORDER BY rowid DESC LIMIT ?) ORDER BY rowid",
                    (last_n,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM messages ORDER BY rowid").fetchall()
            return [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "metadata": json.loads(r["metadata"]),
                }
                for r in rows
            ]

    def clear_messages(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM messages")
            conn.commit()

    # ------------------------------------------------------------------
    # Key-Value
    # ------------------------------------------------------------------

    def save_kv(self, key: str, entry_dict: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, ttl, created_at) VALUES (?, ?, ?, ?)",
                (
                    key,
                    json.dumps(entry_dict["value"]),
                    entry_dict.get("ttl"),
                    entry_dict["created_at"],
                ),
            )
            conn.commit()

    def load_kv(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM kv WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            return {
                "key": row["key"],
                "value": json.loads(row["value"]),
                "ttl": row["ttl"],
                "created_at": row["created_at"],
            }

    def delete_kv(self, key: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def list_kv_keys(self) -> List[str]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT key FROM kv ORDER BY key").fetchall()
            return [r["key"] for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
