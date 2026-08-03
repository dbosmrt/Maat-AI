"""
Settings models for Ma'at Legal AI.

Defines Pydantic models for user settings API.
"""

from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


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


class AvailableModelsResponse(BaseModel):
    """Response with available LLM models."""

    chat_models: Dict[str, str]
    embedding_models: Dict[str, str]


class ChangePasswordRequest(BaseModel):
    """Request to change password."""

    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


class DeleteAccountRequest(BaseModel):
    """Request to delete account."""

    password: str
    confirmation: str = Field(..., pattern=r"^DELETE$", description="Must be 'DELETE' to confirm")
