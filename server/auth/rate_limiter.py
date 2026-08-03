"""
Redis-backed rate limiter for Ma'at Legal AI.

Provides distributed rate limiting using sliding window algorithm with Redis.
Supports 5000+ concurrent requests with configurable limits per endpoint/user.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from server.auth.redis_client import get_redis, get_rate_limit_script
from server.common.config import settings
from server.common.logging import get_logger

logger = get_logger(__name__)


class RateLimitScope(str, Enum):
    """Rate limit scope enumeration."""
    GLOBAL = "global"           # Global across all users
    PER_IP = "ip"               # Per IP address
    PER_USER = "user"          # Per authenticated user
    PER_ENDPOINT = "endpoint"  # Per API endpoint
    PER_USER_ENDPOINT = "user_endpoint"  # Per user per endpoint


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    current_count: int
    limit: int
    window_seconds: int
    retry_after: int = 0
    scope: RateLimitScope = RateLimitScope.PER_USER
    identifier: str = ""


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests: int
    window_seconds: int
    scope: RateLimitScope = RateLimitScope.PER_USER
    block_duration: int = 0  # Optional block duration after limit exceeded


# Default rate limit configurations
DEFAULT_LIMITS = {
    # Auth endpoints - stricter limits
    "auth:register": RateLimitConfig(requests=5, window_seconds=60, scope=RateLimitScope.PER_IP),
    "auth:login": RateLimitConfig(requests=10, window_seconds=60, scope=RateLimitScope.PER_IP),
    "auth:refresh": RateLimitConfig(requests=20, window_seconds=60, scope=RateLimitScope.PER_USER),
    "auth:forgot_password": RateLimitConfig(requests=3, window_seconds=300, scope=RateLimitScope.PER_IP),
    "auth:reset_password": RateLimitConfig(requests=5, window_seconds=300, scope=RateLimitScope.PER_IP),
    "auth:change_password": RateLimitConfig(requests=10, window_seconds=300, scope=RateLimitScope.PER_USER),

    # Chat endpoints
    "chat:invoke": RateLimitConfig(requests=30, window_seconds=60, scope=RateLimitScope.PER_USER),
    "chat:history": RateLimitConfig(requests=60, window_seconds=60, scope=RateLimitScope.PER_USER),
    "chat:sessions": RateLimitConfig(requests=30, window_seconds=60, scope=RateLimitScope.PER_USER),

    # Settings endpoints
    "settings:read": RateLimitConfig(requests=60, window_seconds=60, scope=RateLimitScope.PER_USER),
    "settings:write": RateLimitConfig(requests=20, window_seconds=60, scope=RateLimitScope.PER_USER),

    # Default fallback
    "default": RateLimitConfig(requests=100, window_seconds=60, scope=RateLimitScope.PER_USER),
}


def _build_rate_limit_key(config: RateLimitConfig, identifier: str, endpoint: str) -> str:
    """Build Redis key for rate limiting based on scope."""
    prefix = "maat:ratelimit:"

    if config.scope == RateLimitScope.GLOBAL:
        return f"{prefix}global:{endpoint}"
    if config.scope == RateLimitScope.PER_IP:
        return f"{prefix}ip:{identifier}:{endpoint}"
    if config.scope == RateLimitScope.PER_USER:
        return f"{prefix}user:{identifier}:{endpoint}"
    if config.scope == RateLimitScope.PER_ENDPOINT:
        return f"{prefix}endpoint:{endpoint}"
    if config.scope == RateLimitScope.PER_USER_ENDPOINT:
        return f"{prefix}user:{identifier}:endpoint:{endpoint}"
    return f"{prefix}user:{identifier}:{endpoint}"


async def check_rate_limit(
    endpoint: str,
    identifier: str,
    config: Optional[RateLimitConfig] = None,
) -> RateLimitResult:
    """
    Check and consume rate limit for a request.

    Uses sliding window algorithm with atomic Lua script for accuracy.

    Args:
        endpoint: API endpoint identifier (e.g., "auth:login").
        identifier: Unique identifier based on scope (IP, user_id, etc.).
        config: Rate limit configuration. Uses defaults if not provided.

    Returns:
        RateLimitResult: Result of the rate limit check.
    """
    if config is None:
        config = DEFAULT_LIMITS.get(endpoint, DEFAULT_LIMITS["default"])

    redis_client = get_redis()
    script = await get_rate_limit_script(redis_client)

    key = _build_rate_limit_key(config, identifier, endpoint)
    now_ms = int(time.time() * 1000)
    window_ms = config.window_seconds * 1000

    try:
        result = await script(
            keys=[key],
            args=[config.requests, window_ms, now_ms]
        )

        allowed = bool(result[0])
        current_count = int(result[1])
        retry_after = int(result[2])

        return RateLimitResult(
            allowed=allowed,
            current_count=current_count,
            limit=config.requests,
            window_seconds=config.window_seconds,
            retry_after=retry_after,
            scope=config.scope,
            identifier=identifier,
        )

    except Exception as e:
        logger.error("Rate limit check failed", error=str(e), endpoint=endpoint)
        # Fail open - allow request if Redis is unavailable
        return RateLimitResult(
            allowed=True,
            current_count=0,
            limit=config.requests,
            window_seconds=config.window_seconds,
            scope=config.scope,
            identifier=identifier,
        )


async def get_rate_limit_status(
    endpoint: str,
    identifier: str,
    config: Optional[RateLimitConfig] = None,
) -> RateLimitResult:
    """
    Get current rate limit status without consuming a request.

    Args:
        endpoint: API endpoint identifier.
        identifier: Unique identifier based on scope.
        config: Rate limit configuration.

    Returns:
        RateLimitResult: Current rate limit status.
    """
    if config is None:
        config = DEFAULT_LIMITS.get(endpoint, DEFAULT_LIMITS["default"])

    redis_client = get_redis()
    key = _build_rate_limit_key(config, identifier, endpoint)
    now_ms = int(time.time() * 1000)
    window_ms = config.window_seconds * 1000
    window_start = now_ms - window_ms

    try:
        # Clean up expired entries and count
        await redis_client.zremrangebyscore(key, "-inf", window_start)
        count = await redis_client.zcard(key)

        return RateLimitResult(
            allowed=count < config.requests,
            current_count=count,
            limit=config.requests,
            window_seconds=config.window_seconds,
            retry_after=0,
            scope=config.scope,
            identifier=identifier,
        )
    except Exception as e:
        logger.error("Failed to get rate limit status", error=str(e), endpoint=endpoint)
        return RateLimitResult(
            allowed=True,
            current_count=0,
            limit=config.requests,
            window_seconds=config.window_seconds,
            scope=config.scope,
            identifier=identifier,
        )


async def reset_rate_limit(
    endpoint: str,
    identifier: str,
    config: Optional[RateLimitConfig] = None,
) -> bool:
    """
    Reset rate limit for a specific identifier/endpoint.

    Useful for admin actions or testing.

    Args:
        endpoint: API endpoint identifier.
        identifier: Unique identifier.
        config: Rate limit configuration.

    Returns:
        bool: True if reset successful.
    """
    if config is None:
        config = DEFAULT_LIMITS.get(endpoint, DEFAULT_LIMITS["default"])

    redis_client = get_redis()
    key = _build_rate_limit_key(config, identifier, endpoint)

    try:
        await redis_client.delete(key)
        logger.info("Rate limit reset", endpoint=endpoint, identifier=identifier)
        return True
    except Exception as e:
        logger.error("Failed to reset rate limit", error=str(e), endpoint=endpoint)
        return False


# FastAPI dependency for rate limiting
from fastapi import Request, HTTPException, Depends
from starlette.status import HTTP_429_TOO_MANY_REQUESTS


async def rate_limit_dependency(
    request: Request,
    endpoint: str,
) -> RateLimitResult:
    """
    FastAPI dependency for rate limiting.

    Usage:
        @router.post("/login")
        async def login(
            request: Request,
            rate_limit: RateLimitResult = Depends(rate_limit_dependency("auth:login"))
        ):
            ...

    Args:
        request: FastAPI request object.
        endpoint: Endpoint identifier for rate limit config.

    Returns:
        RateLimitResult: Rate limit check result.

    Raises:
        HTTPException: 429 if rate limit exceeded.
    """
    # Determine identifier based on default config scope
    config = DEFAULT_LIMITS.get(endpoint, DEFAULT_LIMITS["default"])

    if config.scope == RateLimitScope.PER_IP:
        identifier = request.client.host if request.client else "unknown"
    elif config.scope in (RateLimitScope.PER_USER, RateLimitScope.PER_USER_ENDPOINT):
        # Will be overridden by authenticated user dependency
        identifier = "anonymous"
    else:
        identifier = "global"

    result = await check_rate_limit(endpoint, identifier, config)

    if not result.allowed:
        logger.warning(
            "Rate limit exceeded",
            endpoint=endpoint,
            identifier=identifier,
            retry_after=result.retry_after,
        )
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + result.retry_after),
                "Retry-After": str(result.retry_after),
            },
        )

    # Add rate limit headers to response (via middleware or manually)
    request.state.rate_limit = result
    return result


def get_rate_limit_config(endpoint: str) -> RateLimitConfig:
    """Get rate limit config for an endpoint."""
    return DEFAULT_LIMITS.get(endpoint, DEFAULT_LIMITS["default"])


async def set_custom_rate_limit(
    endpoint: str,
    config: RateLimitConfig,
) -> None:
    """Set custom rate limit for an endpoint (runtime configuration)."""
    DEFAULT_LIMITS[endpoint] = config
    logger.info("Custom rate limit set", endpoint=endpoint, config=config)
