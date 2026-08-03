"""
Settings service for Ma'at Legal AI.

Provides user preferences, model configuration, and account management.
"""

from server.settings.models import (
    AvailableModelsResponse,
    ChangePasswordRequest,
    DeleteAccountRequest,
    UserSettingsResponse,
    UserSettingsUpdate,
)
from server.settings.router import router

__all__ = [
    "router",
    "AvailableModelsResponse",
    "ChangePasswordRequest",
    "DeleteAccountRequest",
    "UserSettingsResponse",
    "UserSettingsUpdate",
]
