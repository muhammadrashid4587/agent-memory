"""agent-memory — persistent memory layer for LLM agents."""

__version__ = "0.1.0"

from agent_memory.conversation import ConversationHistory
from agent_memory.kv import KeyValueStore
from agent_memory.memory import AgentMemory
from agent_memory.models import Conversation, KVEntry, Memory, Message, SearchResult
from agent_memory.vector import VectorStore

__all__ = [
    "AgentMemory",
    "VectorStore",
    "ConversationHistory",
    "KeyValueStore",
    "Memory",
    "SearchResult",
    "Message",
    "Conversation",
    "KVEntry",
]
