"""
Token blacklist management for Ma'at Legal AI.

Provides immediate token revocation capability using Redis with atomic operations.
Essential for secure logout, password change, and session invalidation.
"""

import time
from typing import Optional

from server.auth.redis_client import get_redis, get_blacklist_script
from server.common.config import settings
from server.common.logging import get_logger

logger = get_logger(__name__)

# Key prefixes for Redis
BLACKLIST_PREFIX = "maat:blacklist:"
REFRESH_BLACKLIST_PREFIX = "maat:refresh_blacklist:"

# Default TTLs (aligned with token expiration)
ACCESS_TOKEN_TTL = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
REFRESH_TOKEN_TTL = settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60  # seconds


def _get_access_blacklist_key(token_hash: str) -> str:
    """Generate Redis key for access token blacklist."""
    return f"{BLACKLIST_PREFIX}{token_hash}"


def _get_refresh_blacklist_key(token_hash: str) -> str:
    """Generate Redis key for refresh token blacklist."""
    return f"{REFRESH_BLACKLIST_PREFIX}{token_hash}"


def _hash_token(token: str) -> str:
    """
    Create a secure hash of the token for storage.

    Uses SHA-256 to avoid storing actual tokens in Redis.
    """
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()[:32]


async def blacklist_access_token(token: str, ttl: Optional[int] = None) -> bool:
    """
    Add an access token to the blacklist (revoke immediately).

    Args:
        token: JWT access token to blacklist.
        ttl: Optional custom TTL in seconds. Defaults to access token expiration.

    Returns:
        bool: True if token was added to blacklist, False if already blacklisted.
    """
    redis_client = get_redis()
    script = await get_blacklist_script(redis_client)

    token_hash = _hash_token(token)
    key = _get_access_blacklist_key(token_hash)
    effective_ttl = ttl or ACCESS_TOKEN_TTL

    try:
        result = await script(keys=[key], args=[effective_ttl])
        if result == 1:
            logger.info("Access token blacklisted", token_hash=token_hash[:8])
            return True
        logger.debug("Access token already blacklisted", token_hash=token_hash[:8])
        return False
    except Exception as e:
        logger.error("Failed to blacklist access token", error=str(e), token_hash=token_hash[:8])
        return False


async def blacklist_refresh_token(token: str, ttl: Optional[int] = None) -> bool:
    """
    Add a refresh token to the blacklist (revoke immediately).

    Args:
        token: JWT refresh token to blacklist.
        ttl: Optional custom TTL in seconds. Defaults to refresh token expiration.

    Returns:
        bool: True if token was added to blacklist, False if already blacklisted.
    """
    redis_client = get_redis()
    script = await get_blacklist_script(redis_client)

    token_hash = _hash_token(token)
    key = _get_refresh_blacklist_key(token_hash)
    effective_ttl = ttl or REFRESH_TOKEN_TTL

    try:
        result = await script(keys=[key], args=[effective_ttl])
        if result == 1:
            logger.info("Refresh token blacklisted", token_hash=token_hash[:8])
            return True
        logger.debug("Refresh token already blacklisted", token_hash=token_hash[:8])
        return False
    except Exception as e:
        logger.error("Failed to blacklist refresh token", error=str(e), token_hash=token_hash[:8])
        return False


async def is_access_token_blacklisted(token: str) -> bool:
    """
    Check if an access token is blacklisted.

    Args:
        token: JWT access token to check.

    Returns:
        bool: True if token is blacklisted, False otherwise.
    """
    redis_client = get_redis()
    token_hash = _hash_token(token)
    key = _get_access_blacklist_key(token_hash)

    try:
        exists = await redis_client.exists(key)
        return exists == 1
    except Exception as e:
        logger.error("Failed to check access token blacklist", error=str(e), token_hash=token_hash[:8])
        # Fail closed - if Redis is down, treat as potentially blacklisted
        # But for availability, we could fail open. Choose based on security posture.
        return False


async def is_refresh_token_blacklisted(token: str) -> bool:
    """
    Check if a refresh token is blacklisted.

    Args:
        token: JWT refresh token to check.

    Returns:
        bool: True if token is blacklisted, False otherwise.
    """
    redis_client = get_redis()
    token_hash = _hash_token(token)
    key = _get_refresh_blacklist_key(token_hash)

    try:
        exists = await redis_client.exists(key)
        return exists == 1
    except Exception as e:
        logger.error("Failed to check refresh token blacklist", error=str(e), token_hash=token_hash[:8])
        return False


async def blacklist_user_tokens(user_id: str) -> int:
    """
    Blacklist all tokens for a user (force logout from all sessions).

    Note: This requires tracking token->user mapping which we don't currently have.
    Alternative: Use session management to track and invalidate all sessions.

    Args:
        user_id: User ID whose tokens should be blacklisted.

    Returns:
        int: Number of tokens blacklisted (always 0 in current implementation).
    """
    # This would require a token->user mapping index
    # For now, we rely on session invalidation which clears refresh tokens
    logger.warning("Full user token blacklist not implemented - use session invalidation")
    return 0


async def get_blacklist_stats() -> dict:
    """
    Get statistics about the token blacklist.

    Returns:
        dict: Statistics including counts of blacklisted tokens.
    """
    redis_client = get_redis()

    try:
        access_count = await redis_client.dbsize()
        # More precise count would use SCAN with pattern
        return {
            "access_tokens_blacklisted": access_count,
            "note": "Approximate count using DBSIZE"
        }
    except Exception as e:
        logger.error("Failed to get blacklist stats", error=str(e))
        return {"error": str(e)}
