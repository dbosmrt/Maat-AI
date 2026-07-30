"""
Chat service for Ma'at Legal AI.

Provides endpoints for chat session and message management.
"""

from server.chat.router import router
from server.chat.service import ChatService

__all__ = ["router", "ChatService"]