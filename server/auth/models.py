"""
Pydantic models for authentication API.

Defines request/response schemas for auth endpoints.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field



# Request Models


class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password (min 8 chars)")
    full_name: str = Field(..., min_length=1, max_length=255, description="Full name")


class LoginRequest(BaseModel):
    """User login request."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str = Field(..., description="Refresh token")


class ForgotPasswordRequest(BaseModel):
    """Password reset request."""

    email: EmailStr = Field(..., description="User email address")


class ResetPasswordRequest(BaseModel):
    """Password reset confirmation request."""

    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password (min 8 chars)")


class ChangePasswordRequest(BaseModel):
    """Change password request (authenticated user)."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password (min 8 chars)")


# Response Models

class UserResponse(BaseModel):
    """User response model (safe fields only)."""

    id: str
    email: EmailStr
    full_name: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token expiration in seconds")


class AuthResponse(BaseModel):
    """Combined auth response with user and tokens."""

    user: UserResponse
    tokens: TokenResponse


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str



# Token Payload Models

class TokenPayload(BaseModel):
    """JWT token payload structure."""

    sub: str  # user_id
    exp: int  # expiration timestamp
    type: str  # "access" | "refresh" | "password_reset"



# Available Models Response



class AvailableModelsResponse(BaseModel):
    """Response with available LLM models."""

    chat_models: dict[str, str]
    embedding_models: dict[str, str]