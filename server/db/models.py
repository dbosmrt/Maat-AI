"""
Beanie ODM document models for Ma'at Legal AI.

Defines MongoDB document models for User, ChatSession, ChatMessage, and UserSettings.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from beanie import Document, Indexed
from pydantic import BaseModel, EmailStr, Field
from beanie import PydanticObjectId


class UserSettings(Document):
    """User preferences and settings stored as embedded document."""

    class SettingsConfig:
        name = "user_settings"

    user_id: PydanticObjectId = Field(..., description="Reference to User document")
    theme: Literal["light", "dark", "system"] = Field(default="system", description="UI theme preference")
    preferred_chat_model: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1",
        description="Preferred LLM model for chat"
    )
    preferred_embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        description="Preferred embedding model"
    )
    temperature: float = Field(default=0.6, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=8192, ge=1, le=32768, description="Max tokens for LLM response")
    language: str = Field(default="en", description="Preferred language")
    notifications_enabled: bool = Field(default=True, description="Email/push notifications enabled")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        indexes = [
            "user_id",
        ]


class User(Document):
    """User account document."""

    email: EmailStr = Indexed(unique=True, email=True)
    password_hash: str = Field(..., description="Bcrypt hashed password")
    full_name: str = Field(..., min_length=1, max_length=255)
    is_active: bool = Field(default=True, description="Account is active")
    is_verified: bool = Field(default=False, description="Email verified")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = Field(default=None, description="Last successful login")

    # Settings embedded or referenced
    settings: Optional[UserSettings] = Field(default=None, description="User preferences")

    class Settings:
        name = "users"
        indexes = [
            "email",
            "is_active",
            "created_at",
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "full_name": "John Doe",
                "is_active": True,
                "is_verified": True,
            }
        }


class ChatSession(Document):
    """Chat session document."""

    user_id: PydanticObjectId = Indexed()
    title: str = Field(default="New Conversation", max_length=500)
    memory_summary: str = Field(default="", description="Memory summary for context")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = Field(default=0, ge=0)
    is_archived: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Settings:
        name = "chat_sessions"
        indexes = [
            "user_id",
            "created_at",
            "updated_at",
            [("user_id", 1), ("updated_at", -1)],
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "title": "Property Law Question",
                "memory_summary": "User asked about property inheritance",
            }
        }


class MessageRole(str, Enum):
    """Message role enumeration."""

    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"


class ChatMessage(Document):
    """Individual chat message document."""

    session_id: PydanticObjectId = Indexed()
    user_id: PydanticObjectId = Indexed()
    role: MessageRole = Field(..., description="Message role: human, ai, or system")
    content: str = Field(..., min_length=1, description="Message content")
    law_domain: Optional[str] = Field(default=None, description="Detected legal domain")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chat_messages"
        indexes = [
            "session_id",
            "user_id",
            "created_at",
            [("session_id", 1), ("created_at", 1)],
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "507f1f77bcf86cd799439012",
                "user_id": "507f1f77bcf86cd799439011",
                "role": "human",
                "content": "What are the inheritance laws in India?",
                "law_domain": "Civil",
            }
        }



# Pydantic models for API requests/responses (not stored in DB)


class UserSettingsUpdate(BaseModel):
    """Request model for updating user settings."""

    theme: Optional[Literal["light", "dark", "system"]] = None
    preferred_chat_model: Optional[str] = None
    preferred_embedding_model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=32768)
    language: Optional[str] = None
    notifications_enabled: Optional[bool] = None


class UserSettingsResponse(BaseModel):
    """Response model for user settings."""

    theme: Literal["light", "dark", "system"]
    preferred_chat_model: str
    preferred_embedding_model: str
    temperature: float
    max_tokens: int
    language: str
    notifications_enabled: bool


class ChatSessionCreate(BaseModel):
    """Request model for creating a chat session."""

    title: Optional[str] = Field(default="New Conversation", max_length=500)


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


class ChatMessageCreate(BaseModel):
    """Request model for creating a chat message."""

    role: MessageRole
    content: str = Field(..., min_length=1)
    law_domain: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    """Response model for chat history."""

    session_id: str
    messages: List[ChatMessageResponse]


# Available models for user selection
AVAILABLE_CHAT_MODELS = {
    "nemotron3super": "nvidia/llama-3.3-nemotron-super-49b-v1",
    "glm5_2": "z-ai/glm-5.2",
    "sarvam_m": "sarvamai/sarvam-m",
    "minimax_m3": "minimaxai/minimax-m3",
    "nemotron_ultra": "nvidia/nemotron-3-ultra",
    "llama3_1_70b": "meta/llama-3.1-70b-instruct",
}

AVAILABLE_EMBEDDING_MODELS = {
    "nv_embedqa_e5_v5": "nvidia/nv-embedqa-e5-v5",
    "nv_embed_v2": "nvidia/nv-embed-v2",
}
