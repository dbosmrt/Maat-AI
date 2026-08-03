"""
Redis client management for Ma'at Legal AI authentication system.

Provides async Redis client with connection pooling for high-concurrency scenarios.
Designed to handle 5000+ concurrent requests with proper connection management.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from server.common.config import settings
from server.common.logging import get_logger

logger = get_logger(__name__)

# Global Redis client and pool instances
_redis_client: Optional[redis.Redis] = None  # pylint: disable=invalid-name
_redis_pool: Optional[ConnectionPool] = None  # pylint: disable=invalid-name


async def init_redis() -> redis.Redis:
    """
    Initialize Redis connection pool and client.

    Returns:
        redis.Redis: Initialized Redis client instance.

    Raises:
        RuntimeError: If Redis initialization fails.
    """
    global _redis_client, _redis_pool

    if _redis_client is not None:
        logger.warning("Redis already initialized, returning existing connection")
        return _redis_client

    logger.info(f"Initializing Redis connection pool to {settings.REDIS_URI}")

    try:
        # Create connection pool optimized for high concurrency
        # 5000 concurrent requests -> pool size of 100-200 with proper timeouts
        _redis_pool = ConnectionPool.from_url(
            settings.REDIS_URI,
            max_connections=200,  # Support 5k concurrent with connection reuse
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30,
            decode_responses=True,
        )

        _redis_client = redis.Redis(connection_pool=_redis_pool)

        # Test connection
        await _redis_client.ping()
        logger.info("Redis connection pool initialized successfully")

        return _redis_client

    except Exception as e:
        logger.critical(f"Failed to initialize Redis: {e}")
        await close_redis()
        raise RuntimeError(f"Redis initialization failed: {e}") from e


async def close_redis() -> None:
    """Close Redis connection pool and clean up resources."""
    global _redis_client, _redis_pool

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None

    if _redis_pool is not None:
        await _redis_pool.disconnect()
        _redis_pool = None

    logger.info("Redis connection pool closed")


def get_redis() -> redis.Redis:
    """
    Get the initialized Redis client instance.

    Returns:
        redis.Redis: The Redis client instance.

    Raises:
        RuntimeError: If Redis has not been initialized.
    """
    if _redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis_client


@asynccontextmanager
async def redis_lifespan():
    """
    Context manager for Redis lifecycle.

    Usage:
        async with redis_lifespan():
            # Redis operations here
            pass
        # Redis automatically closed
    """
    try:
        await init_redis()
        yield get_redis()
    finally:
        await close_redis()


async def get_redis_dependency() -> redis.Redis:
    """
    FastAPI dependency for Redis injection.

    Returns:
        redis.Redis: The Redis client instance.
    """
    return get_redis()


# Lua scripts for atomic operations (loaded once, executed many times)
# These are defined here to be registered on first use

# Token blacklist check and add (atomic)
BLACKLIST_CHECK_AND_ADD_SCRIPT = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local exists = redis.call('EXISTS', key)
if exists == 1 then
    return 0  -- Already blacklisted
end
redis.call('SETEX', key, ttl, '1')
return 1  -- Successfully added
"""

# Rate limit check and increment (atomic sliding window)
RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local window_start = now - window

-- Remove expired entries
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count current requests
local count = redis.call('ZCARD', key)

if count >= limit then
    -- Get TTL of oldest entry for retry-after
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = 0
    if #oldest > 0 then
        retry_after = math.ceil((tonumber(oldest[2]) + window - now) / 1000)
    end
    return {0, count, retry_after}
end

-- Add current request
redis.call('ZADD', key, now, now .. '-' .. math.random(1000000))
redis.call('EXPIRE', key, math.ceil(window / 1000) + 1)

return {1, count + 1, 0}
"""

# Failed login attempt tracking (atomic)
FAILED_LOGIN_SCRIPT = """
local key = KEYS[1]
local max_attempts = tonumber(ARGV[1])
local lockout_duration = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local attempts = tonumber(redis.call('GET', key) or '0')
attempts = attempts + 1
redis.call('SET', key, attempts, 'EX', lockout_duration)

if attempts >= max_attempts then
    return {0, attempts, lockout_duration}
end
return {1, attempts, 0}
"""

# Session management (atomic add/remove)
SESSION_ADD_SCRIPT = """
local sessions_key = KEYS[1]
local session_key = KEYS[2]
local user_id = ARGV[1]
local device_info = ARGV[2]
local ttl = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Add to user sessions set
redis.call('ZADD', sessions_key, now, session_key)
redis.call('EXPIRE', sessions_key, ttl)

-- Store session details
redis.call('HSET', session_key, 'user_id', user_id, 'device_info', device_info, 'created_at', now)
redis.call('EXPIRE', session_key, ttl)

return 1
"""

SESSION_REMOVE_SCRIPT = """
local sessions_key = KEYS[1]
local session_key = KEYS[2]

redis.call('ZREM', sessions_key, session_key)
redis.call('DEL', session_key)

return 1
"""

# Initialize Lua scripts on first use
_scripts_initialized = False  # pylint: disable=invalid-name
_blacklist_script = None  # pylint: disable=invalid-name
_rate_limit_script = None  # pylint: disable=invalid-name
_failed_login_script = None  # pylint: disable=invalid-name
_session_add_script = None  # pylint: disable=invalid-name
_session_remove_script = None  # pylint: disable=invalid-name


async def _ensure_scripts_loaded(redis_client: redis.Redis) -> None:
    """Ensure Lua scripts are loaded (called lazily on first use)."""
    global _scripts_initialized, _blacklist_script, _rate_limit_script
    global _failed_login_script, _session_add_script, _session_remove_script

    if not _scripts_initialized:
        _blacklist_script = redis_client.register_script(BLACKLIST_CHECK_AND_ADD_SCRIPT)
        _rate_limit_script = redis_client.register_script(RATE_LIMIT_SCRIPT)
        _failed_login_script = redis_client.register_script(FAILED_LOGIN_SCRIPT)
        _session_add_script = redis_client.register_script(SESSION_ADD_SCRIPT)
        _session_remove_script = redis_client.register_script(SESSION_REMOVE_SCRIPT)
        _scripts_initialized = True


async def get_blacklist_script(redis_client: redis.Redis):
    await _ensure_scripts_loaded(redis_client)
    return _blacklist_script


async def get_rate_limit_script(redis_client: redis.Redis):
    await _ensure_scripts_loaded(redis_client)
    return _rate_limit_script


async def get_failed_login_script(redis_client: redis.Redis):
    await _ensure_scripts_loaded(redis_client)
    return _failed_login_script


async def get_session_add_script(redis_client: redis.Redis):
    await _ensure_scripts_loaded(redis_client)
    return _session_add_script


async def get_session_remove_script(redis_client: redis.Redis):
    await _ensure_scripts_loaded(redis_client)
    return _session_remove_script
