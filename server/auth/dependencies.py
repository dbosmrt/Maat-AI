"""
FastAPI authentication dependencies.

Provides dependency injection for current user, authentication, authorization,
token blacklist checking, and session validation.
"""

from typing import Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.auth.rate_limiter import RateLimitScope, check_rate_limit, get_rate_limit_config
from server.auth.security import (
    decode_token,
    get_token_user_id,
    get_token_type,
    verify_token_type,
    is_token_blacklisted,
)
from server.auth.session_manager import get_session, update_session_activity
from server.common.exceptions import AuthenticationError, AuthorizationError
from server.common.logging import get_logger
from server.db.models import User

logger = get_logger(__name__)

# HTTP Bearer token scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """
    Extract and validate user ID from JWT access token.

    Args:
        credentials: HTTP Authorization credentials (Bearer token).

    Returns:
        str: The authenticated user ID.

    Raises:
        AuthenticationError: If token is missing, invalid, expired, or blacklisted.
    """
    if not credentials:
        logger.warning("Authentication failed: Missing authorization header")
        raise AuthenticationError(
            "Missing authorization header",
            details={"header": "Authorization"}
        )

    token = credentials.credentials

    # Validate token type
    if not verify_token_type(token, "access"):
        logger.warning("Authentication failed: Invalid token type")
        raise AuthenticationError(
            "Invalid token type",
            details={"expected": "access", "received": get_token_type(token)}
        )

    # Check if token is blacklisted (revoked)
    if await is_token_blacklisted(token):
        logger.warning("Authentication failed: Token revoked")
        raise AuthenticationError(
            "Token has been revoked",
            details={"code": "TOKEN_REVOKED"}
        )

    # Extract user ID
    user_id = get_token_user_id(token)
    if not user_id:
        logger.warning("Authentication failed: Invalid token payload")
        raise AuthenticationError(
            "Invalid token payload",
            details={"field": "sub"}
        )

    return user_id


async def get_current_user(
    user_id: str = Depends(get_current_user_id),
) -> User:
    """
    Get the current authenticated user document.

    Args:
        user_id: User ID from validated token.

    Returns:
        User: The authenticated user document.

    Raises:
        AuthenticationError: If user not found or inactive.
    """
    user = await User.get(user_id)
    if not user:
        logger.warning("Authentication failed: User not found", user_id=user_id)
        raise AuthenticationError(
            "User not found",
            details={"user_id": user_id}
        )

    if not user.is_active:
        logger.warning("Authentication failed: User account inactive", user_id=user_id)
        raise AuthenticationError(
            "Account is deactivated",
            details={"user_id": user_id}
        )

    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    """
    Get current user ensuring they are active.

    Args:
        user: Current user from get_current_user.

    Returns:
        User: The active user.

    Raises:
        AuthorizationError: If user is not active.
    """
    if not user.is_active:
        raise AuthorizationError("Inactive user account")
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[User]:
    """
    Optionally get current user if authenticated.

    Does not raise on missing/invalid token - returns None instead.

    Args:
        credentials: Optional HTTP Authorization credentials.

    Returns:
        Optional[User]: The authenticated user or None.
    """
    if not credentials:
        return None

    try:
        user_id = await get_current_user_id(credentials)
        return await User.get(user_id)
    except AuthenticationError:
        return None


def require_verified_user(user: User = Depends(get_current_active_user)) -> User:
    """
    Require user to have verified email.

    Args:
        user: Current active user.

    Returns:
        User: The verified user.

    Raises:
        AuthorizationError: If user email is not verified.
    """
    if not user.is_verified:
        raise AuthorizationError(
            "Email verification required",
            details={"field": "is_verified"}
        )
    return user


# Rate limiting dependencies

async def get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxies."""
    # Check for forwarded headers (when behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    return request.client.host if request.client else "unknown"


def auth_rate_limit_dependency(endpoint: str):
    """
    Factory for rate limiting dependency for auth endpoints.

    Args:
        endpoint: Endpoint identifier for rate limit config.

    Returns:
        Dependency function that checks rate limit.
    """
    async def rate_limit_check(request: Request) -> None:
        config = get_rate_limit_config(endpoint)
        identifier = await get_client_ip(request)

        # For user-scoped limits, we need the user ID
        if config.scope in (RateLimitScope.PER_USER, RateLimitScope.PER_USER_ENDPOINT):
            # Try to get user from token
            credentials = bearer_scheme(request)
            if credentials:
                try:
                    user_id = get_token_user_id(credentials.credentials)
                    if user_id:
                        identifier = user_id
                except Exception:
                    pass  # Fall back to IP

        await check_rate_limit(endpoint, identifier, config)

    return rate_limit_check


# Session validation dependencies

async def get_current_session(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """
    Get current session ID from request.

    Extracts session ID from token JTI claim.

    Args:
        credentials: Optional HTTP Authorization credentials.

    Returns:
        Optional[str]: Session ID if found.
    """
    if credentials:
        token = credentials.credentials
        from server.auth.security import get_token_jti
        return get_token_jti(token)
    return None


async def validate_session(
    session_id: str,
    user_id: str,
) -> bool:
    """
    Validate that a session exists and belongs to the user.

    Args:
        session_id: Session ID to validate.
        user_id: User ID for ownership check.

    Returns:
        bool: True if session is valid.
    """
    session = await get_session(session_id)
    if not session:
        return False
    return session.user_id == user_id


async def require_session(
    session_id: str = Depends(get_current_session),
    user: User = Depends(get_current_active_user),
) -> str:
    """
    Require a valid session for the current user.

    Args:
        session_id: Session ID from request.
        user: Current authenticated user.

    Returns:
        str: Valid session ID.

    Raises:
        AuthenticationError: If session is invalid or doesn't belong to user.
    """
    if not session_id:
        raise AuthenticationError("No active session")

    is_valid = await validate_session(session_id, str(user.id))
    if not is_valid:
        raise AuthenticationError("Invalid or expired session")

    # Update last activity
    await update_session_activity(session_id)

    return session_id


# Correlation ID extraction
async def get_correlation_id(request: Request) -> str:
    """Extract or generate correlation ID for request tracing."""
    return request.headers.get("X-Correlation-ID", "")


# Combined dependency for authenticated requests with rate limiting
async def authenticated_request(
    request: Request,
    user: User = Depends(get_current_active_user),
    correlation_id: str = Depends(get_correlation_id),
) -> dict:
    """
    Combined dependency for authenticated requests.

    Provides user, request context, and correlation ID in one dependency.

    Returns:
        dict: Context with user, request, and correlation_id.
    """
    return {
        "user": user,
        "request": request,
        "correlation_id": correlation_id,
        "user_id": str(user.id),
    }
