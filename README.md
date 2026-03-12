# agent-memory

Persistent memory layer for LLM agents. Provides **vector store** (semantic search), **conversation history** (windowed + summarized), and **key-value state** — all with pluggable backends.

## Features

- **Vector Store** — Store text memories and retrieve them via TF-IDF cosine similarity search. Pure Python, no numpy/sklearn required.
- **Conversation History** — Sliding window with token-aware truncation. Supports `max_tokens` budgeting.
- **Key-Value State** — Typed get/set/delete with optional TTL (time-to-live) expiry.
- **Pluggable Backends** — In-memory (ephemeral) or SQLite (persistent, WAL mode).
- **Thread-Safe** — All stores are fully thread-safe.
- **Zero Heavy Dependencies** — Only requires `pydantic`. No numpy, sklearn, or FAISS needed.

## Installation

```bash
pip install agent-memory
```

Or install from source:

```bash
git clone https://github.com/muhammadrashid4587/agent-memory.git
cd agent-memory
pip install -e ".[dev]"
```

## Quick Start

```python
from agent_memory import AgentMemory

mem = AgentMemory(backend="sqlite", path="agent.db")

# Store memories
mem.store("The user prefers dark mode", metadata={"type": "preference"})
mem.store("User's name is Alice", metadata={"type": "identity"})

# Semantic search
results = mem.search("what does the user like?", top_k=3)

# Conversation history
mem.add_message("user", "Hello!")
mem.add_message("assistant", "Hi there!")
history = mem.get_history(last_n=10)

# Key-value state
mem.set("last_tool", "web_search", ttl=3600)
mem.get("last_tool")  # -> "web_search"
```

## Backends

### In-Memory (default)

```python
mem = AgentMemory(backend="memory")
```

Fast and ephemeral. Data is lost when the process exits. Great for testing.

### SQLite

```python
mem = AgentMemory(backend="sqlite", path="agent.db")
```

Persistent storage using SQLite with WAL journal mode for good concurrent-read performance. Data survives across sessions.

## API Reference

### AgentMemory

The main facade class that combines all three stores.

| Method | Description |
|--------|-------------|
| `store(text, metadata=None, memory_id=None)` | Store a memory for semantic retrieval |
| `search(query, top_k=5)` | Search memories by semantic similarity |
| `delete_memory(memory_id)` | Delete a specific memory |
| `memory_count()` | Count stored memories |
| `add_message(role, content, metadata=None)` | Add a conversation message |
| `get_history(last_n=None, max_tokens=None)` | Retrieve conversation history |
| `clear_history()` | Clear all conversation messages |
| `set(key, value, ttl=None)` | Set a key-value pair |
| `get(key, default=None)` | Get a value by key |
| `delete_key(key)` | Delete a key-value pair |
| `close()` | Release backend resources |

### Token-Aware Windowing

```python
mem = AgentMemory(backend="memory", max_tokens=4096)

# Messages are automatically truncated to fit the token budget
for i in range(100):
    mem.add_message("user", f"Message {i} with some content...")

# Only the most recent messages fitting within 4096 tokens are returned
history = mem.get_history()
```

### TTL (Time-to-Live)

```python
mem.set("session_token", "abc123", ttl=3600)  # expires in 1 hour
mem.set("permanent", "value")                  # never expires
```

### Using Individual Stores

You can also use the stores directly for more control:

```python
from agent_memory import VectorStore, ConversationHistory, KeyValueStore
from agent_memory.backends import MemoryBackend

backend = MemoryBackend()

vectors = VectorStore(backend)
vectors.add("Some important fact")
results = vectors.search("fact")

convo = ConversationHistory(backend, max_tokens=2048)
convo.add_message("user", "Hello")

kv = KeyValueStore(backend)
kv.set("key", "value")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=agent_memory --cov-report=term-missing
```

## License

MIT
