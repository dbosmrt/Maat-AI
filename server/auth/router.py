"""
Authentication router for Ma'at Legal AI.

Provides endpoints for user registration, login, token refresh, password management,
session management, email verification, API key management, and user profile management.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, EmailStr, Field

from server.auth.dependencies import (
    auth_rate_limit_dependency,
    get_current_active_user,
    get_current_session,
    get_current_user,
    require_session,
    require_verified_user,
)
from server.auth.models import (
    AuthResponse,
    AvailableModelsResponse,
    MessageResponse,
    TokenResponse,
    UserResponse,
)
from server.auth.rate_limiter import RateLimitConfig, RateLimitScope, set_custom_rate_limit
from server.auth.security import (
    create_password_reset_token,
    create_token_pair,
    hash_password,
    refresh_token_pair,
    validate_password_strength,
    verify_password,
)
from server.auth.security_events import (
    log_account_locked,
    log_all_sessions_revoked,
    log_login_failure,
    log_login_success,
    log_logout,
    log_password_change,
    log_security_event_kwargs,
    log_session_revoked,
    SecurityEventType,
    SecurityEventSeverity,
)
from server.auth.session_manager import (
    create_session,
    list_user_sessions,
    revoke_all_sessions,
    revoke_session,
)
from server.common.exceptions import (
    AuthenticationError,
    ConflictError,
    ResourceNotFoundError,
    ValidationError,
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


class EmailVerificationRequest(BaseModel):
    """Email verification request."""

    token: str


class SessionRevokeRequest(BaseModel):
    """Session revocation request."""

    session_id: str


class ApiKeyCreateRequest(BaseModel):
    """API key creation request."""

    name: str = Field(..., min_length=1, max_length=100, description="Human-readable name for the API key")
    expires_in_days: Optional[int] = Field(default=365, ge=1, le=3650, description="Expiration in days")


class ApiKeyResponse(BaseModel):
    """API key response (includes key only on creation)."""

    id: str
    name: str
    key: str  # Only returned on creation
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime] = None


class SessionInfoResponse(BaseModel):
    """Session information response."""

    session_id: str
    device_info: str
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_current: bool


class SessionsListResponse(BaseModel):
    """List of user sessions."""

    sessions: list[SessionInfoResponse]


# Auth Endpoints


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(auth_rate_limit_dependency("auth:register"))],
)
async def register(request: RegisterRequest, http_request: Request):
    """
    Register a new user account.

    - **email**: User's email address (unique)
    - **password**: Password (min 8 chars, must meet strength requirements)
    - **full_name**: User's full name
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    user_agent = http_request.headers.get("user-agent", "")

    logger.info("User registration attempted", email=request.email, ip=client_ip)

    # Validate password strength
    is_valid, error_msg = validate_password_strength(request.password)
    if not is_valid:
        logger.warning("Registration failed: Weak password", email=request.email, reason=error_msg)
        raise ValidationError(error_msg or "Password does not meet requirements")

    # Check if user already exists
    existing_user = await User.find_one({"email": request.email.lower()})
    if existing_user:
        await log_login_failure(request.email, client_ip, user_agent, "Email already registered")
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

    # Create initial session
    _session = await create_session(str(user.id), http_request)

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

    await log_login_success(str(user.id), user.email, client_ip, user_agent)

    return AuthResponse(user=user_response, tokens=token_response)


@router.post(
    "/login",
    response_model=AuthResponse,
    dependencies=[Depends(auth_rate_limit_dependency("auth:login"))],
)
async def login(request: LoginRequest, http_request: Request):
    """
    Authenticate user and return access/refresh tokens.

    - **email**: User's email address
    - **password**: User's password
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    user_agent = http_request.headers.get("user-agent", "")

    logger.info("User login attempted", email=request.email, ip=client_ip)

    # Find user
    user = await User.find_one({"email": request.email.lower()})
    if not user:
        await log_login_failure(request.email, client_ip, user_agent, "User not found")
        logger.warning("Login failed: User not found", email=request.email)
        raise AuthenticationError("Invalid credentials")

    # Verify password
    if not verify_password(request.password, user.password_hash):
        await log_login_failure(request.email, client_ip, user_agent, "Invalid password")
        logger.warning("Login failed: Invalid password", email=request.email, user_id=str(user.id))
        raise AuthenticationError("Invalid credentials")

    # Check if user is active
    if not user.is_active:
        await log_login_failure(request.email, client_ip, user_agent, "Account deactivated")
        logger.warning("Login failed: Account deactivated", user_id=str(user.id))
        raise AuthenticationError("Account is deactivated")

    # Update last login
    user.last_login_at = datetime.utcnow()
    await user.save()

    logger.info("User logged in successfully", user_id=str(user.id), email=request.email)

    # Create token pair
    tokens = create_token_pair(str(user.id))

    # Create session
    _session = await create_session(str(user.id), http_request)

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

    await log_login_success(str(user.id), user.email, client_ip, user_agent)

    return AuthResponse(user=user_response, tokens=token_response)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    dependencies=[Depends(auth_rate_limit_dependency("auth:refresh"))],
)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token (with rotation).

    - **refresh_token**: Valid refresh token

    Returns new access token AND new refresh token (rotation).
    """
    logger.debug("Token refresh attempted")

    # Validate refresh token and create new token pair (rotation)
    new_tokens = await refresh_token_pair(request.refresh_token)
    if not new_tokens:
        logger.warning("Token refresh failed: Invalid or expired refresh token")
        raise AuthenticationError("Invalid or expired refresh token")

    return TokenResponse(
        access_token=new_tokens["access_token"],
        refresh_token=new_tokens["refresh_token"],
        expires_in=new_tokens["access_expires_in"],
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
async def logout(
    http_request: Request,
    user: User = Depends(get_current_active_user),
    session_id: str = Depends(require_session),
):
    """
    Logout current user (revokes current session and tokens).
    """
    logger.info("User logged out", user_id=str(user.id))

    # Revoke current session
    await revoke_session(session_id, str(user.id))

    # Tokens are stateless but we could add to blacklist if needed
    # For now, session revocation handles the logout

    await log_logout(str(user.id), user.email, http_request.client.host if http_request.client else "unknown")

    return MessageResponse(message="Logged out successfully")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(request: PasswordResetRequest, http_request: Request):
    """
    Request password reset email.

    - **email**: User's email address

    Note: In production, this would send an email with a reset link.
    For now, it logs the reset token (check server logs for token).
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
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

    await log_security_event_kwargs(
        event_type=SecurityEventType.PASSWORD_RESET_REQUEST,
        user_id=str(user.id),
        email=user.email,
        ip_address=client_ip,
        message=f"Password reset requested for {user.email}",
    )

    return MessageResponse(message="If the email exists, a reset link has been sent")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(request: PasswordResetConfirmRequest, http_request: Request):
    """
    Reset password using reset token.

    - **token**: Password reset token from email
    - **new_password**: New password (min 8 chars, must meet strength requirements)
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    logger.info("Password reset attempted")

    # Validate new password strength
    is_valid, error_msg = validate_password_strength(request.new_password)
    if not is_valid:
        raise ValidationError(error_msg or "Password does not meet requirements")

    # Decode and validate reset token
    from server.auth.security import decode_token
    payload = decode_token(request.token)
    if not payload:
        await log_security_event_kwargs(
            event_type=SecurityEventType.PASSWORD_RESET_FAILURE,
            email=request.email,
            ip_address=client_ip,
            message="Invalid or expired reset token used",
            severity=SecurityEventSeverity.WARNING,
        )
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

    # Revoke all sessions (force re-login)
    revoked_count = await revoke_all_sessions(str(user.id))

    logger.info("Password reset successful", user_id=user_id)

    await log_password_change(str(user.id), user.email, client_ip)
    await log_all_sessions_revoked(str(user.id), user.email, revoked_count, client_ip)

    return MessageResponse(message="Password has been reset successfully")


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    request: PasswordChangeRequest,
    user: User = Depends(get_current_active_user),
    http_request: Request = None,
):
    """
    Change password for authenticated user.

    - **current_password**: Current password
    - **new_password**: New password (min 8 chars, must meet strength requirements)
    """
    client_ip = http_request.client.host if http_request and http_request.client else "unknown"
    logger.info("Password change requested", user_id=str(user.id))

    # Validate new password strength
    is_valid, error_msg = validate_password_strength(request.new_password)
    if not is_valid:
        raise ValidationError(error_msg or "Password does not meet requirements")

    # Verify current password
    if not verify_password(request.current_password, user.password_hash):
        logger.warning("Password change failed: Incorrect current password", user_id=str(user.id))
        await log_security_event_kwargs(
            event_type=SecurityEventType.PASSWORD_RESET_FAILURE,
            user_id=str(user.id),
            email=user.email,
            ip_address=client_ip,
            message="Password change failed: incorrect current password",
            severity=SecurityEventSeverity.WARNING,
        )
        raise AuthenticationError("Current password is incorrect")

    # Hash and update new password
    user.password_hash = hash_password(request.new_password)
    await user.save()

    # Revoke all other sessions (keep current)
    current_session = await get_current_session() if http_request else None
    revoked_count = await revoke_all_sessions(str(user.id), except_session_id=current_session)

    logger.info("Password changed successfully", user_id=str(user.id))

    await log_password_change(str(user.id), user.email, client_ip)
    await log_all_sessions_revoked(str(user.id), user.email, revoked_count, client_ip)

    return MessageResponse(message="Password changed successfully")


# Email Verification Endpoints

@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(request: EmailVerificationRequest):
    """
    Verify user email with verification token.

    - **token**: Email verification token
    """
    logger.info("Email verification attempted")

    from server.auth.security import decode_token
    payload = decode_token(request.token)
    if not payload:
        raise AuthenticationError("Invalid or expired verification token")

    if payload.get("type") != "email_verification":
        raise AuthenticationError("Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token payload")

    user = await User.get(user_id)
    if not user:
        raise ResourceNotFoundError("User", user_id)

    if user.is_verified:
        return MessageResponse(message="Email already verified")

    user.is_verified = True
    await user.save()

    logger.info("Email verified successfully", user_id=user_id)
    return MessageResponse(message="Email verified successfully")


@router.post("/resend-verification", response_model=MessageResponse)
async def resend_verification_email(user: User = Depends(get_current_active_user)):
    """
    Resend email verification link to current user.
    """
    if user.is_verified:
        return MessageResponse(message="Email already verified")

    # Generate verification token
    from server.auth.security import create_access_token
    verification_token = create_access_token(str(user.id), expires_delta=timedelta(hours=24))
    # Add type claim manually (or create dedicated function)
    # In production: send email with verification link

    logger.info("Verification email resent", user_id=str(user.id), token=verification_token)
    return MessageResponse(message="Verification email sent")


# Session Management Endpoints

@router.get("/sessions", response_model=SessionsListResponse)
async def list_sessions(
    user: User = Depends(get_current_active_user),
    current_session: str = Depends(get_current_session),
):
    """
    List all active sessions for the current user.
    """
    sessions = await list_user_sessions(str(user.id), current_session)

    session_responses = [
        SessionInfoResponse(
            session_id=s.session_id,
            device_info=s.device_info,
            created_at=datetime.fromtimestamp(s.created_at),
            last_activity=datetime.fromtimestamp(s.last_activity),
            ip_address=s.ip_address,
            user_agent=s.user_agent,
            is_current=s.is_current,
        )
        for s in sessions
    ]

    return SessionsListResponse(sessions=session_responses)


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
async def revoke_session_endpoint(
    session_id: str,
    user: User = Depends(get_current_active_user),
    http_request: Request = None,
):
    """
    Revoke a specific session (logout from a specific device).
    """
    client_ip = http_request.client.host if http_request and http_request.client else "unknown"

    success = await revoke_session(session_id, str(user.id))
    if not success:
        raise ResourceNotFoundError("Session", session_id)

    await log_session_revoked(str(user.id), user.email, session_id, client_ip)

    return MessageResponse(message="Session revoked successfully")


@router.delete("/sessions", response_model=MessageResponse)
async def revoke_all_sessions_endpoint(
    user: User = Depends(get_current_active_user),
    http_request: Request = None,
    current_session: str = Depends(get_current_session),
):
    """
    Revoke all sessions except current (logout from all other devices).
    """
    client_ip = http_request.client.host if http_request and http_request.client else "unknown"

    revoked_count = await revoke_all_sessions(str(user.id), except_session_id=current_session)

    await log_all_sessions_revoked(str(user.id), user.email, revoked_count, client_ip)

    return MessageResponse(message=f"Revoked {revoked_count} other sessions")


# API Key Management (for service-to-service auth)

@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: ApiKeyCreateRequest,
    user: User = Depends(require_verified_user),
):
    """
    Create a new API key for service-to-service authentication.

    - **name**: Human-readable name for the API key
    - **expires_in_days**: Expiration in days (default: 365, max: 10 years)
    """
    from server.auth.security import create_access_token

    # Generate API key
    api_key = f"maat_{secrets.token_urlsafe(32)}"
    key_hash = hash_password(api_key)  # Store hashed

    expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

    # Store in user settings or separate collection
    # For simplicity, store in user metadata
    if not user.metadata:
        user.metadata = {}
    if "api_keys" not in user.metadata:
        user.metadata["api_keys"] = []

    key_id = secrets.token_urlsafe(16)
    user.metadata["api_keys"].append({
        "id": key_id,
        "name": request.name,
        "key_hash": key_hash,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": expires_at.isoformat(),
        "last_used_at": None,
    })
    await user.save()

    logger.info("API key created", user_id=str(user.id), key_name=request.name, key_id=key_id)

    return ApiKeyResponse(
        id=key_id,
        name=request.name,
        key=api_key,  # Only returned once!
        created_at=datetime.utcnow(),
        expires_at=expires_at,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(user: User = Depends(require_verified_user)):
    """List all API keys for the current user (keys are hashed, not shown)."""
    api_keys = user.metadata.get("api_keys", []) if user.metadata else []

    return [
        ApiKeyResponse(
            id=k["id"],
            name=k["name"],
            key="",  # Don't return actual key
            created_at=datetime.fromisoformat(k["created_at"]),
            expires_at=datetime.fromisoformat(k["expires_at"]) if k.get("expires_at") else None,
            last_used_at=datetime.fromisoformat(k["last_used_at"]) if k.get("last_used_at") else None,
        )
        for k in api_keys
    ]


@router.delete("/api-keys/{key_id}", response_model=MessageResponse)
async def revoke_api_key(key_id: str, user: User = Depends(require_verified_user)):
    """Revoke an API key."""
    api_keys = user.metadata.get("api_keys", []) if user.metadata else []
    api_keys = [k for k in api_keys if k["id"] != key_id]

    if len(api_keys) == (len(user.metadata.get("api_keys", [])) if user.metadata else 0):
        raise ResourceNotFoundError("API Key", key_id)

    if user.metadata:
        user.metadata["api_keys"] = api_keys
    await user.save()

    logger.info("API key revoked", user_id=str(user.id), key_id=key_id)
    return MessageResponse(message="API key revoked successfully")


# Settings Endpoints (also available under /api/v1/settings)

@router.get("/available-models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get available LLM and embedding models."""
    return AvailableModelsResponse(
        chat_models=AVAILABLE_CHAT_MODELS,
        embedding_models=AVAILABLE_EMBEDDING_MODELS,
    )


# Admin/Utility Endpoints

@router.get("/security-events")
async def get_security_events(
    limit: int = 100,
    offset: int = 0,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    user: User = Depends(require_verified_user),
):
    """
    Get security events for the current user (admin only in production).
    """
    from server.auth.security_events import get_security_events as get_events

    et = SecurityEventType(event_type) if event_type else None
    sev = SecurityEventSeverity(severity) if severity else None

    events = await get_events(limit=limit, offset=offset, event_type=et, user_id=str(user.id), severity=sev)
    return [e.to_dict() for e in events]
