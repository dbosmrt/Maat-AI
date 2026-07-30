"""
Authentication router for Ma'at Legal AI.

Provides endpoints for user registration, login, token refresh, password management,
and user profile management.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, EmailStr, Field

from server.auth.dependencies import get_current_active_user
from server.auth.models import (
    AuthResponse,
    AvailableModelsResponse,
    MessageResponse,
    TokenResponse,
    UserResponse,
)
from server.auth.security import (
    create_access_token,
    create_password_reset_token,
    create_refresh_token,
    create_token_pair,
    decode_token,
    hash_password,
    refresh_access_token,
    verify_password,
)
from server.common.exceptions import (
    AuthenticationError,
    ConflictError,
    ResourceNotFoundError,
)
from server.common.logging import get_logger
from server.db.models import User, UserSettings
from server.db.models import AVAILABLE_CHAT_MODELS, AVAILABLE_EMBEDDING_MODELS

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])



# Request Models

class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(..., min_length=1, max_length=255)


class LoginRequest(BaseModel):
    """User login request."""

    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request."""

    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


class PasswordResetRequest(BaseModel):
    """Password reset request (forgot password)."""

    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation request."""

    token: str
    new_password: str = Field(..., min_length=8, max_length=128)



# Auth Endpoints


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user account.

    - **email**: User's email address (unique)
    - **password**: Password (min 8 chars)
    - **full_name**: User's full name
    """
    logger.info("User registration attempted", email=request.email)

    # Check if user already exists
    existing_user = await User.find_one({"email": request.email.lower()})
    if existing_user:
        logger.warning("Registration failed: Email already registered", email=request.email)
        raise ConflictError("User", request.email)

    # Hash password
    password_hash = hash_password(request.password)

    # Create user
    user = User(
        email=request.email.lower(),
        password_hash=password_hash,
        full_name=request.full_name,
        is_active=True,
        is_verified=False,  # Require email verification in production
    )

    # Create default settings
    user_settings = UserSettings(user_id=user.id)
    await user_settings.insert()

    user.settings = user_settings
    await user.insert()

    logger.info("User registered successfully", user_id=str(user.id), email=request.email)

    # Create token pair
    tokens = create_token_pair(str(user.id))

    # Prepare response
    user_response = UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )

    token_response = TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["access_expires_in"],
    )

    return AuthResponse(user=user_response, tokens=token_response)


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return access/refresh tokens.

    - **email**: User's email address
    - **password**: User's password
    """
    logger.info("User login attempted", email=request.email)

    # Find user
    user = await User.find_one({"email": request.email.lower()})
    if not user:
        logger.warning("Login failed: User not found", email=request.email)
        raise AuthenticationError("Invalid credentials")

    # Verify password
    if not verify_password(request.password, user.password_hash):
        logger.warning("Login failed: Invalid password", email=request.email, user_id=str(user.id))
        raise AuthenticationError("Invalid credentials")

    # Check if user is active
    if not user.is_active:
        logger.warning("Login failed: Account deactivated", user_id=str(user.id))
        raise AuthenticationError("Account is deactivated")

    # Update last login
    user.last_login_at = datetime.utcnow()
    await user.save()

    logger.info("User logged in successfully", user_id=str(user.id), email=request.email)

    # Create token pair
    tokens = create_token_pair(str(user.id))

    # Prepare response
    user_response = UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )

    token_response = TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["access_expires_in"],
    )

    return AuthResponse(user=user_response, tokens=token_response)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.

    - **refresh_token**: Valid refresh token
    """
    logger.debug("Token refresh attempted")

    # Validate refresh token and create new access token
    new_access_token = refresh_access_token(request.refresh_token)
    if not new_access_token:
        logger.warning("Token refresh failed: Invalid or expired refresh token")
        raise AuthenticationError("Invalid or expired refresh token")

    # Get user ID from refresh token for new refresh token
    from server.auth.security import get_token_user_id
    user_id = get_token_user_id(request.refresh_token)

    # Create new token pair for security
    tokens = create_token_pair(user_id)

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["access_expires_in"],
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user's profile.
    """
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(user: User = Depends(get_current_active_user)):
    """
    Logout current user (client-side token invalidation).

    Note: JWT tokens are stateless. This endpoint exists for client-side
    cleanup and audit logging. Server-side token revocation would require
    a token blacklist (Redis).
    """
    logger.info("User logged out", user_id=str(user.id))
    return MessageResponse(message="Logged out successfully")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(request: PasswordResetRequest):
    """
    Request password reset email.

    - **email**: User's email address

    Note: In production, this would send an email with a reset link.
    For now, it logs the reset token (check server logs for token).
    """
    logger.info("Password reset requested", email=request.email)

    user = await User.find_one({"email": request.email.lower()})
    if not user:
        # Don't reveal if email exists - always return success
        logger.info("Password reset requested for non-existent email", email=request.email)
        return MessageResponse(message="If the email exists, a reset link has been sent")

    # Generate reset token
    reset_token = create_password_reset_token(str(user.id))

    # In production: send email with reset link containing token
    # For development: log the token
    logger.info(
        "Password reset token generated",
        user_id=str(user.id),
        email=request.email,
        reset_token=reset_token,
    )

    return MessageResponse(message="If the email exists, a reset link has been sent")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(request: PasswordResetConfirmRequest):
    """
    Reset password using reset token.

    - **token**: Password reset token from email
    - **new_password**: New password (min 8 chars)
    """
    logger.info("Password reset attempted")

    # Decode and validate reset token
    payload = decode_token(request.token)
    if not payload:
        raise AuthenticationError("Invalid or expired reset token")

    if payload.get("type") != "password_reset":
        raise AuthenticationError("Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token payload")

    # Find user
    user = await User.get(user_id)
    if not user:
        raise ResourceNotFoundError("User", user_id)

    # Hash new password
    user.password_hash = hash_password(request.new_password)
    await user.save()

    logger.info("Password reset successful", user_id=user_id)
    return MessageResponse(message="Password has been reset successfully")


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    request: PasswordChangeRequest,
    user: User = Depends(get_current_active_user),
):
    """
    Change password for authenticated user.

    - **current_password**: Current password
    - **new_password**: New password (min 8 chars)
    """
    logger.info("Password change requested", user_id=str(user.id))

    # Verify current password
    if not verify_password(request.current_password, user.password_hash):
        logger.warning("Password change failed: Incorrect current password", user_id=str(user.id))
        raise AuthenticationError("Current password is incorrect")

    # Hash and update new password
    user.password_hash = hash_password(request.new_password)
    await user.save()

    logger.info("Password changed successfully", user_id=str(user.id))
    return MessageResponse(message="Password changed successfully")



# Settings Endpoints (also available under /api/v1/settings)


@router.get("/available-models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get available LLM and embedding models."""
    return AvailableModelsResponse(
        chat_models=AVAILABLE_CHAT_MODELS,
        embedding_models=AVAILABLE_EMBEDDING_MODELS,
    )