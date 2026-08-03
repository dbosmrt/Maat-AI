"""
Chat service models for Ma'at Legal AI.

Defines Pydantic models for chat session and message API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role enumeration."""

    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"


class ChatSessionBase(BaseModel):
    """Base chat session model."""

    title: str = Field(default="New Conversation", max_length=500)
    is_archived: bool = False


class ChatSessionCreate(ChatSessionBase):
    """Request model for creating a chat session."""


class ChatSessionUpdate(BaseModel):
    """Request model for updating a chat session."""

    title: Optional[str] = Field(default=None, max_length=500)
    is_archived: Optional[bool] = None


class ChatSessionResponse(BaseModel):
    """Response model for chat session."""

    id: str
    user_id: str
    title: str
    memory_summary: str
    message_count: int
    is_archived: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatMessageBase(BaseModel):
    """Base chat message model."""

    role: MessageRole
    content: str = Field(..., min_length=1)
    law_domain: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessageCreate(ChatMessageBase):
    """Request model for creating a chat message."""


class ChatMessageResponse(BaseModel):
    """Response model for chat message."""

    id: str
    session_id: str
    user_id: str
    role: MessageRole
    content: str
    law_domain: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    """Response model for chat history with messages."""

    session_id: str
    messages: List[ChatMessageResponse]


class ChatRequest(BaseModel):
    """Chat request model (user query)."""

    query: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = Field(default=None, description="Existing session ID or 'new'")


class ChatResponse(BaseModel):
    """Chat response model."""

    session_id: str
    generation: str
    law_domain: str


class SessionListResponse(BaseModel):
    """Response model for session list."""

    sessions: List[ChatSessionResponse]
    total: int
    page: int
    page_size: int
