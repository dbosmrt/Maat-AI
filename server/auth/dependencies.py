"""
FastAPI authentication dependencies.

Provides dependency injection for current user, authentication, and authorization.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server.auth.security import (
    decode_token,
    get_token_user_id,
    verify_token_type,
)
from server.common.config import settings
from server.common.exceptions import AuthenticationError, AuthorizationError
from server.common.logging import get_logger
from server.db.connection import get_database
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
        AuthenticationError: If token is missing, invalid, or expired.
    """
    if not credentials:
        logger.warning("Authentication failed: Missing authorization header")
        raise AuthenticationError("Missing authorization header", details={"header": "Authorization"})

    token = credentials.credentials

    # Validate token type
    if not verify_token_type(token, "access"):
        logger.warning("Authentication failed: Invalid token type")
        raise AuthenticationError("Invalid token type", details={"expected": "access"})

    # Extract user ID
    user_id = get_token_user_id(token)
    if not user_id:
        logger.warning("Authentication failed: Invalid token payload")
        raise AuthenticationError("Invalid token payload", details={"field": "sub"})

    return user_id


async def get_current_user(
    user_id: str = Depends(get_current_user_id),
    database=Depends(get_database),
) -> User:
    """
    Get the current authenticated user document.

    Args:
        user_id: User ID from validated token.
        database: MongoDB database instance.

    Returns:
        User: The authenticated user document.

    Raises:
        AuthenticationError: If user not found or inactive.
    """
    user = await User.get(user_id)
    if not user:
        logger.warning("Authentication failed: User not found", user_id=user_id)
        raise AuthenticationError("User not found", details={"user_id": user_id})

    if not user.is_active:
        logger.warning("Authentication failed: User account inactive", user_id=user_id)
        raise AuthenticationError("Account is deactivated", details={"user_id": user_id})

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
    database=Depends(get_database),
) -> Optional[User]:
    """
    Optionally get current user if authenticated.

    Does not raise on missing/invalid token - returns None instead.

    Args:
        credentials: Optional HTTP Authorization credentials.
        database: MongoDB database instance.

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
        raise AuthorizationError("Email verification required", details={"field": "is_verified"})
    return user
