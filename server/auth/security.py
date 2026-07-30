"""
Authentication security utilities for Ma'at Legal AI.

Provides password hashing, JWT token creation/validation, and token utilities.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt
from passlib.context import CryptContext

from server.common.config import settings
from server.common.exceptions import AuthenticationError
from server.common.logging import get_logger

logger = get_logger(__name__)

# Password hashing context - bcrypt with appropriate rounds
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token configuration
ALGORITHM = settings.JWT_ALGORITHM
SECRET_KEY = settings.JWT_SECRET_KEY
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS
PASSWORD_RESET_TOKEN_EXPIRE_HOURS = settings.PASSWORD_RESET_TOKEN_EXPIRE_HOURS


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plaintext password.

    Returns:
        str: Hashed password.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a hashed password.

    Args:
        plain_password: Plaintext password to verify.
        hashed_password: Stored hashed password.

    Returns:
        bool: True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User ID to encode in token.
        expires_delta: Optional custom expiration timedelta.

    Returns:
        str: Encoded JWT access token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {"sub": user_id, "exp": expire, "type": "access", "iat": datetime.now(timezone.utc)}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug("Access token created", user_id=user_id, expires_at=expire.isoformat())
    return encoded_jwt


def create_refresh_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: User ID to encode in token.
        expires_delta: Optional custom expiration timedelta.

    Returns:
        str: Encoded JWT refresh token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {"sub": user_id, "exp": expire, "type": "refresh", "iat": datetime.now(timezone.utc)}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug("Refresh token created", user_id=user_id, expires_at=expire.isoformat())
    return encoded_jwt


def create_password_reset_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a password reset token.

    Args:
        user_id: User ID to encode in token.
        expires_delta: Optional custom expiration timedelta.

    Returns:
        str: Encoded JWT password reset token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=PASSWORD_RESET_TOKEN_EXPIRE_HOURS)

    to_encode = {
        "sub": user_id,
        "exp": expire,
        "type": "password_reset",
        "iat": datetime.now(timezone.utc),
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug("Password reset token created", user_id=user_id, expires_at=expire.isoformat())
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string.

    Returns:
        Optional[dict]: Decoded payload if valid, None if invalid/expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.JWTError as e:
        logger.warning("Invalid token", error=str(e))
        return None


def get_token_user_id(token: str) -> Optional[str]:
    """
    Extract user ID from token payload.

    Args:
        token: JWT token string.

    Returns:
        Optional[str]: User ID (sub claim) if present and valid.
    """
    payload = decode_token(token)
    if payload and isinstance(payload.get("sub"), str):
        return payload["sub"]
    return None


def get_token_type(token: str) -> Optional[str]:
    """
    Extract token type from token payload.

    Args:
        token: JWT token string.

    Returns:
        Optional[str]: Token type (access, refresh, password_reset) if present.
    """
    payload = decode_token(token)
    if payload and isinstance(payload.get("type"), str):
        return payload["type"]
    return None


def verify_token_type(token: str, expected_type: str) -> bool:
    """
    Verify token matches expected type.

    Args:
        token: JWT token string.
        expected_type: Expected token type.

    Returns:
        bool: True if token type matches, False otherwise.
    """
    token_type = get_token_type(token)
    return token_type == expected_type


def get_token_expiration(token: str) -> Optional[datetime]:
    """
    Get token expiration datetime.

    Args:
        token: JWT token string.

    Returns:
        Optional[datetime]: Expiration datetime if token is valid.
    """
    payload = decode_token(token)
    if payload and isinstance(payload.get("exp"), int):
        return datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    return None


def is_token_expired(token: str) -> bool:
    """
    Check if a token is expired without raising an exception.

    Args:
        token: JWT token string.

    Returns:
        bool: True if expired or invalid, False if valid and not expired.
    """
    expiration = get_token_expiration(token)
    if expiration is None:
        return True
    return datetime.now(timezone.utc) >= expiration


def create_token_pair(user_id: str) -> dict[str, str]:
    """
    Create both access and refresh tokens for a user.

    Args:
        user_id: User ID for token subject.

    Returns:
        dict: Dictionary with access_token, refresh_token, and metadata.
    """
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "access_expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "refresh_expires_in": REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
    }


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """
    Create a new access token from a valid refresh token.

    Args:
        refresh_token: Valid refresh token.

    Returns:
        Optional[str]: New access token if refresh token is valid, None otherwise.
    """
    if not verify_token_type(refresh_token, "refresh"):
        return None

    user_id = get_token_user_id(refresh_token)
    if not user_id:
        return None

    if is_token_expired(refresh_token):
        return None

    return create_access_token(user_id)