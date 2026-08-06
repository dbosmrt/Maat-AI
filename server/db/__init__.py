"""
Database module for Ma'at Legal AI.

Provides MongoDB connection management and Beanie ODM models.
"""

from server.db.connection import (
    close_database,
    init_database,
    get_database,
    init_beanie_models,
)
from server.db.models import (
    User,
    UserSettings,
    ChatSession,
    ChatMessage,
    MessageRole,
    UserSettingsUpdate,
    UserSettingsResponse,
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionResponse,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatHistoryResponse,
    AVAILABLE_CHAT_MODELS,
    AVAILABLE_EMBEDDING_MODELS,
)

__all__ = [
    # Connection
    "init_database",
    "close_database",
    "get_database",
    "init_beanie_models",
    # Models
    "User",
    "UserSettings",
    "ChatSession",
    "ChatMessage",
    "MessageRole",
    # Pydantic models
    "UserSettingsUpdate",
    "UserSettingsResponse",
    "ChatSessionCreate",
    "ChatSessionUpdate",
    "ChatSessionResponse",
    "ChatMessageCreate",
    "ChatMessageResponse",
    "ChatHistoryResponse",
    # Constants
    "AVAILABLE_CHAT_MODELS",
    "AVAILABLE_EMBEDDING_MODELS",
]
