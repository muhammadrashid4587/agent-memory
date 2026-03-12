"""Pydantic models for agent-memory."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single memory entry stored in the vector store."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

    class Config:
        json_encoders = {float: lambda v: round(v, 6)}


class SearchResult(BaseModel):
    """A single search result returned by vector search."""

    memory: Memory
    score: float  # cosine similarity in [0, 1]


class Message(BaseModel):
    """A single conversation message."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A conversation consisting of a list of messages."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: List[Message] = Field(default_factory=list)
    summary: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class KVEntry(BaseModel):
    """A key-value entry with optional TTL."""

    key: str
    value: Any
    ttl: Optional[float] = None  # seconds; None = no expiry
    created_at: float = Field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
