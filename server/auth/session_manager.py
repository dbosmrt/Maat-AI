"""
Session management for Ma'at Legal AI.

Tracks active user sessions, enables listing/revoking sessions,
and provides device tracking for security monitoring.
"""

import secrets
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from server.auth.redis_client import get_redis, get_session_add_script, get_session_remove_script
from server.common.config import settings
from server.common.logging import get_logger

logger = get_logger(__name__)

# Key prefixes
SESSIONS_PREFIX = "maat:sessions:"
SESSION_DETAIL_PREFIX = "maat:session:"

# Default session TTL (aligned with refresh token)
DEFAULT_SESSION_TTL = settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60


@dataclass
class SessionInfo:
    """Information about an active session."""
    session_id: str
    user_id: str
    device_info: str
    created_at: int
    last_activity: int
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_current: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "device_info": self.device_info,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_current": self.is_current,
        }


def _get_user_sessions_key(user_id: str) -> str:
    """Get Redis key for user's session set."""
    return f"{SESSIONS_PREFIX}{user_id}"


def _get_session_detail_key(session_id: str) -> str:
    """Get Redis key for session details."""
    return f"{SESSION_DETAIL_PREFIX}{session_id}"


def _generate_session_id() -> str:
    """Generate a secure random session ID."""
    return secrets.token_urlsafe(32)


def _parse_device_info(user_agent: str, ip: str) -> str:
    """Parse user agent into a human-readable device identifier."""
    # Simple parsing - in production, use ua-parser or similar
    ua_lower = user_agent.lower()
    device = "Unknown"
    if "mobile" in ua_lower or "android" in ua_lower or "iphone" in ua_lower:
        device = "Mobile"
    elif "tablet" in ua_lower or "ipad" in ua_lower:
        device = "Tablet"
    elif "windows" in ua_lower:
        device = "Windows"
    elif "macintosh" in ua_lower or "mac os" in ua_lower:
        device = "macOS"
    elif "linux" in ua_lower:
        device = "Linux"

    browser = "Unknown"
    if "chrome" in ua_lower and "edg" not in ua_lower:
        browser = "Chrome"
    elif "firefox" in ua_lower:
        browser = "Firefox"
    elif "safari" in ua_lower and "chrome" not in ua_lower:
        browser = "Safari"
    elif "edg" in ua_lower:
        browser = "Edge"

    return f"{device} - {browser} ({ip})"


async def create_session(
    user_id: str,
    request = None,
    device_info: Optional[str] = None,
    ttl: Optional[int] = None,
) -> SessionInfo:
    """
    Create a new session for a user.

    Args:
        user_id: User ID.
        request: Optional FastAPI request object for extracting IP/User-Agent.
        device_info: Optional pre-parsed device info.
        ttl: Optional custom TTL in seconds.

    Returns:
        SessionInfo: Created session information.
    """
    redis_client = get_redis()
    script = await get_session_add_script(redis_client)

    session_id = _generate_session_id()
    now = int(time.time())
    effective_ttl = ttl or DEFAULT_SESSION_TTL

    # Extract device info from request if provided
    ip_address = None
    user_agent = None
    if request:
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

    if device_info is None:
        device_info = _parse_device_info(user_agent or "", ip_address or "unknown")

    sessions_key = _get_user_sessions_key(user_id)
    session_key = _get_session_detail_key(session_id)

    try:
        await script(
            keys=[sessions_key, session_key],
            args=[user_id, device_info, effective_ttl, now]
        )

        # Store additional metadata
        await redis_client.hset(session_key, mapping={
            "ip_address": ip_address or "",
            "user_agent": user_agent or "",
            "last_activity": str(now),
        })
        await redis_client.expire(session_key, effective_ttl)

        logger.info("Session created", user_id=user_id, session_id=session_id[:8], device=device_info)

        return SessionInfo(
            session_id=session_id,
            user_id=user_id,
            device_info=device_info,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )

    except Exception as e:
        logger.error("Failed to create session", error=str(e), user_id=user_id)
        raise


async def get_session(session_id: str) -> Optional[SessionInfo]:
    """
    Get session details by session ID.

    Args:
        session_id: Session ID.

    Returns:
        Optional[SessionInfo]: Session info if found, None otherwise.
    """
    redis_client = get_redis()
    session_key = _get_session_detail_key(session_id)

    try:
        data = await redis_client.hgetall(session_key)
        if not data:
            return None

        return SessionInfo(
            session_id=session_id,
            user_id=data.get("user_id", ""),
            device_info=data.get("device_info", ""),
            created_at=int(data.get("created_at", 0)),
            last_activity=int(data.get("last_activity", 0)),
            ip_address=data.get("ip_address") or None,
            user_agent=data.get("user_agent") or None,
        )
    except Exception as e:
        logger.error("Failed to get session", error=str(e), session_id=session_id[:8])
        return None


async def update_session_activity(session_id: str) -> bool:
    """
    Update session last activity timestamp.

    Args:
        session_id: Session ID.

    Returns:
        bool: True if updated successfully.
    """
    redis_client = get_redis()
    session_key = _get_session_detail_key(session_id)
    now = int(time.time())

    try:
        await redis_client.hset(session_key, "last_activity", str(now))
        # Extend expiry on activity
        await redis_client.expire(session_key, DEFAULT_SESSION_TTL)
        return True
    except Exception as e:
        logger.error("Failed to update session activity", error=str(e), session_id=session_id[:8])
        return False


async def list_user_sessions(user_id: str, current_session_id: Optional[str] = None) -> List[SessionInfo]:
    """
    List all active sessions for a user.

    Args:
        user_id: User ID.
        current_session_id: Optional current session ID to mark as current.

    Returns:
        List[SessionInfo]: List of active sessions, most recent first.
    """
    redis_client = get_redis()
    sessions_key = _get_user_sessions_key(user_id)

    try:
        # Get session IDs sorted by creation time (newest first)
        session_ids = await redis_client.zrevrange(sessions_key, 0, -1)

        sessions = []
        for session_id in session_ids:
            session = await get_session(session_id)
            if session:
                session.is_current = (session_id == current_session_id)
                sessions.append(session)

        return sessions
    except Exception as e:
        logger.error("Failed to list user sessions", error=str(e), user_id=user_id)
        return []


async def revoke_session(session_id: str, user_id: Optional[str] = None) -> bool:
    """
    Revoke (delete) a specific session.

    Args:
        session_id: Session ID to revoke.
        user_id: Optional user ID for verification.

    Returns:
        bool: True if session was revoked.
    """
    redis_client = get_redis()

    # Verify ownership if user_id provided
    if user_id:
        session = await get_session(session_id)
        if not session or session.user_id != user_id:
            logger.warning("Session revocation denied: ownership mismatch", session_id=session_id[:8])
            return False

    session_key = _get_session_detail_key(session_id)
    sessions_key = _get_user_sessions_key(user_id) if user_id else None

    try:
        if sessions_key:
            script = await get_session_remove_script(redis_client)
            await script(keys=[sessions_key, session_key])
        else:
            # Just delete the session detail
            await redis_client.delete(session_key)

        logger.info("Session revoked", session_id=session_id[:8], user_id=user_id)
        return True
    except Exception as e:
        logger.error("Failed to revoke session", error=str(e), session_id=session_id[:8])
        return False


async def revoke_all_sessions(user_id: str, except_session_id: Optional[str] = None) -> int:
    """
    Revoke all sessions for a user (force logout everywhere).

    Args:
        user_id: User ID.
        except_session_id: Optional session ID to keep active.

    Returns:
        int: Number of sessions revoked.
    """
    redis_client = get_redis()
    sessions_key = _get_user_sessions_key(user_id)

    try:
        session_ids = await redis_client.zrange(sessions_key, 0, -1)
        revoked = 0

        for session_id in session_ids:
            if session_id != except_session_id:
                session_key = _get_session_detail_key(session_id)
                script = await get_session_remove_script(redis_client)
                await script(keys=[sessions_key, session_key])
                revoked += 1

        logger.info("All sessions revoked", user_id=user_id, count=revoked, except_session=except_session_id[:8] if except_session_id else None)
        return revoked
    except Exception as e:
        logger.error("Failed to revoke all sessions", error=str(e), user_id=user_id)
        return 0


async def get_session_count(user_id: str) -> int:
    """
    Get count of active sessions for a user.

    Args:
        user_id: User ID.

    Returns:
        int: Number of active sessions.
    """
    redis_client = get_redis()
    sessions_key = _get_user_sessions_key(user_id)

    try:
        return await redis_client.zcard(sessions_key)
    except Exception as e:
        logger.error("Failed to get session count", error=str(e), user_id=user_id)
        return 0


async def cleanup_expired_sessions() -> int:
    """
    Clean up expired sessions (maintenance task).

    Returns:
        int: Number of sessions cleaned up.
    """
    # Redis handles expiry automatically via TTL
    # This function is for manual cleanup if needed
    logger.info("Session cleanup not needed - Redis handles TTL automatically")
    return 0
