"""
Authentication security utilities for Ma'at Legal AI.

Provides password hashing, JWT token creation/validation, token rotation,
password strength validation, and token utilities.
"""

import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from jose import jwt
from passlib.context import CryptContext

from server.auth.token_blacklist import (
    blacklist_access_token,
    blacklist_refresh_token,
    is_access_token_blacklisted,
    is_refresh_token_blacklisted,
)
from server.common.config import settings
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

# Password strength requirements
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
REQUIRE_UPPERCASE = True
REQUIRE_LOWERCASE = True
REQUIRE_DIGIT = True
REQUIRE_SPECIAL_CHAR = True

# Common weak passwords (extend as needed)
WEAK_PASSWORDS = frozenset([
    "password", "password123", "12345678", "qwerty123", "admin123",
    "welcome123", "letmein", "monkey123", "dragon123", "master123",
])


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


def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """
    Validate password strength against policy.

    Args:
        password: Password to validate.

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message_if_invalid)
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"

    if len(password) > MAX_PASSWORD_LENGTH:
        return False, f"Password must not exceed {MAX_PASSWORD_LENGTH} characters"

    if REQUIRE_UPPERCASE and not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"

    if REQUIRE_LOWERCASE and not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"

    if REQUIRE_DIGIT and not re.search(r"\d", password):
        return False, "Password must contain at least one digit"

    if REQUIRE_SPECIAL_CHAR and not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
        return False, "Password must contain at least one special character"

    # Check against common weak passwords
    if password.lower() in WEAK_PASSWORDS:
        return False, "Password is too common. Please choose a stronger password"

    # Check for sequential characters (e.g., "123", "abc")
    if _has_sequential_chars(password):
        return False, "Password contains sequential characters"

    # Check for repeated characters (e.g., "aaa", "111")
    if _has_repeated_chars(password):
        return False, "Password contains too many repeated characters"

    return True, None


def _has_sequential_chars(password: str, min_length: int = 3) -> bool:
    """Check for sequential characters (ascending or descending)."""
    password_lower = password.lower()
    for i in range(len(password_lower) - min_length + 1):
        substring = password_lower[i:i + min_length]
        # Check ascending (e.g., abc, 123)
        if all(ord(substring[j]) + 1 == ord(substring[j + 1]) for j in range(len(substring) - 1)):
            return True
        # Check descending (e.g., cba, 321)
        if all(ord(substring[j]) - 1 == ord(substring[j + 1]) for j in range(len(substring) - 1)):
            return True
    return False


def _has_repeated_chars(password: str, max_repeats: int = 3) -> bool:
    """Check for repeated characters."""
    for i in range(len(password) - max_repeats + 1):
        if len(set(password[i:i + max_repeats])) == 1:
            return True
    return False


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

    to_encode = {
        "sub": user_id,
        "exp": expire,
        "type": "access",
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(16),  # Unique token identifier
    }
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

    to_encode = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_urlsafe(16),  # Unique token identifier
    }
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
        "jti": secrets.token_urlsafe(16),
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


def get_token_jti(token: str) -> Optional[str]:
    """
    Extract JWT ID (jti) from token payload.

    Args:
        token: JWT token string.

    Returns:
        Optional[str]: JWT ID if present.
    """
    payload = decode_token(token)
    if payload and isinstance(payload.get("jti"), str):
        return payload["jti"]
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


async def is_token_blacklisted(token: str) -> bool:
    """
    Check if a token is blacklisted (revoked).

    Args:
        token: JWT token string.

    Returns:
        bool: True if token is blacklisted.
    """
    token_type = get_token_type(token)
    if token_type == "access":
        return await is_access_token_blacklisted(token)
    if token_type == "refresh":
        return await is_refresh_token_blacklisted(token)
    return False


async def revoke_token(token: str) -> bool:
    """
    Revoke a token by adding to blacklist.

    Args:
        token: JWT token string.

    Returns:
        bool: True if token was revoked.
    """
    token_type = get_token_type(token)
    if token_type == "access":
        return await blacklist_access_token(token)
    if token_type == "refresh":
        return await blacklist_refresh_token(token)
    return False


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


async def refresh_access_token(refresh_token: str) -> Optional[str]:
    """
    Create a new access token from a valid refresh token.
    Implements refresh token rotation: issues new refresh token, invalidates old.

    Args:
        refresh_token: Valid refresh token.

    Returns:
        Optional[str]: New access token if refresh token is valid, None otherwise.
    """
    # Verify it's a refresh token
    if not verify_token_type(refresh_token, "refresh"):
        logger.warning("Token refresh failed: Not a refresh token")
        return None

    # Check if refresh token is blacklisted
    if await is_refresh_token_blacklisted(refresh_token):
        logger.warning("Token refresh failed: Refresh token revoked")
        return None

    user_id = get_token_user_id(refresh_token)
    if not user_id:
        logger.warning("Token refresh failed: Invalid token payload")
        return None

    if is_token_expired(refresh_token):
        logger.warning("Token refresh failed: Refresh token expired")
        return None

    # Rotate refresh token: blacklist old, create new pair
    await blacklist_refresh_token(refresh_token)
    tokens = create_token_pair(user_id)

    logger.info("Token refreshed with rotation", user_id=user_id)
    return tokens["access_token"]


async def refresh_token_pair(refresh_token: str) -> Optional[dict]:
    """
    Create a new token pair (access + refresh) from a valid refresh token.
    Implements full refresh token rotation.

    Args:
        refresh_token: Valid refresh token.

    Returns:
        Optional[dict]: New token pair if valid, None otherwise.
    """
    if not verify_token_type(refresh_token, "refresh"):
        return None

    if await is_refresh_token_blacklisted(refresh_token):
        return None

    user_id = get_token_user_id(refresh_token)
    if not user_id:
        return None

    if is_token_expired(refresh_token):
        return None

    # Rotate: blacklist old refresh token
    await blacklist_refresh_token(refresh_token)

    # Create new token pair
    tokens = create_token_pair(user_id)
    logger.info("Full token pair refreshed with rotation", user_id=user_id)
    return tokens
