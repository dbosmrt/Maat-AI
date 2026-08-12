"""
Security events audit logging for Ma'at Legal AI.

Provides comprehensive audit logging for security-relevant events
including authentication, authorization, and account changes.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List

from server.auth.redis_client import get_redis
from server.common.logging import get_logger

logger = get_logger(__name__)


class SecurityEventType(str, Enum):
    """Types of security events."""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REVOKED = "token_revoked"

    # Registration events
    REGISTRATION = "registration"
    EMAIL_VERIFICATION = "email_verification"

    # Password events
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET_REQUEST = "password_reset_request"
    PASSWORD_RESET_SUCCESS = "password_reset_success"
    PASSWORD_RESET_FAILURE = "password_reset_failure"

    # Account events
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    ACCOUNT_DEACTIVATED = "account_deactivated"
    ACCOUNT_REACTIVATED = "account_reactivated"

    # Session events
    SESSION_CREATED = "session_created"
    SESSION_REVOKED = "session_revoked"
    ALL_SESSIONS_REVOKED = "all_sessions_revoked"

    # MFA events (future)
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    MFA_CHALLENGE = "mfa_challenge"
    MFA_SUCCESS = "mfa_success"
    MFA_FAILURE = "mfa_failure"

    # API key events (future)
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_USED = "api_key_used"

    # Admin events
    ADMIN_ACTION = "admin_action"
    ROLE_CHANGED = "role_changed"
    PERMISSION_CHANGED = "permission_changed"


class SecurityEventSeverity(str, Enum):
    """Severity levels for security events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: SecurityEventType
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    severity: SecurityEventSeverity = SecurityEventSeverity.INFO
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp(self) -> datetime:
        """Get event timestamp (auto-generated)."""
        return datetime.now(timezone.utc)

    @property
    def correlation_id(self) -> Optional[str]:
        """Get correlation ID from metadata."""
        return self.metadata.get("correlation_id")

    @property
    def email(self) -> Optional[str]:
        """Get email from metadata."""
        return self.metadata.get("email")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "email": self.email,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# Redis key for security event log (append-only list)
SECURITY_LOG_KEY = "maat:security:events"
MAX_LOG_ENTRIES = 100000  # Keep last 100k events


async def log_security_event(event: SecurityEvent) -> None:
    """
    Log a security event to both structured logger and Redis.

    Args:
        event: SecurityEvent to log.
    """
    # Log to structured logger
    log_data = event.to_dict()

    if event.severity == SecurityEventSeverity.CRITICAL:
        logger.critical(event.message, **log_data)
    elif event.severity == SecurityEventSeverity.WARNING:
        logger.warning(event.message, **log_data)
    else:
        logger.info(event.message, **log_data)

    # Store in Redis for audit trail
    await _store_event_in_redis(event)


def _build_security_event(
    *,
    event_type: SecurityEventType,
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
    message: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> SecurityEvent:
    """Build a SecurityEvent from keyword arguments."""
    combined_metadata = metadata or {}
    if email:
        combined_metadata["email"] = email
    if correlation_id:
        combined_metadata["correlation_id"] = correlation_id
    return SecurityEvent(
        event_type=event_type,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        severity=severity,
        message=message,
        metadata=combined_metadata,
    )


async def log_security_event_kwargs(
    event_type: SecurityEventType,
    *,
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
    message: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """
    Log a security event using keyword arguments (convenience function).

    Args:
        event_type: Type of security event.
        user_id: User ID.
        email: User email.
        ip_address: Client IP address.
        user_agent: Client user agent.
        severity: Event severity.
        message: Human-readable message.
        metadata: Additional metadata.
        correlation_id: Correlation ID for tracing.
    """
    event = _build_security_event(
        event_type=event_type,
        user_id=user_id,
        email=email,
        ip_address=ip_address,
        user_agent=user_agent,
        severity=severity,
        message=message,
        metadata=metadata,
        correlation_id=correlation_id,
    )
    await log_security_event(event)


async def _store_event_in_redis(event: SecurityEvent) -> None:
    """Store event in Redis list for audit trail."""
    redis_client = get_redis()

    try:
        event_json = event.to_json()
        # Use LPUSH for newest-first ordering
        await redis_client.lpush(SECURITY_LOG_KEY, event_json)
        # Trim to max entries
        await redis_client.ltrim(SECURITY_LOG_KEY, 0, MAX_LOG_ENTRIES - 1)
    except Exception as e:
        # Don't fail the operation if audit logging fails
        logger.error("Failed to store security event in Redis", error=str(e))


def _build_event_filter(
    *,
    limit: int = 100,
    offset: int = 0,
    event_type: Optional[SecurityEventType] = None,
    user_id: Optional[str] = None,
    severity: Optional[SecurityEventSeverity] = None,
    since: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build filter parameters for querying security events."""
    return {
        "limit": limit,
        "offset": offset,
        "event_type": event_type,
        "user_id": user_id,
        "severity": severity,
        "since": since,
    }


async def get_security_events(
    *,
    limit: int = 100,
    offset: int = 0,
    event_type: Optional[SecurityEventType] = None,
    user_id: Optional[str] = None,
    severity: Optional[SecurityEventSeverity] = None,
    since: Optional[datetime] = None,
) -> List[SecurityEvent]:
    """
    Retrieve security events with filtering.

    Args:
        limit: Maximum number of events to return.
        offset: Number of events to skip.
        event_type: Filter by event type.
        user_id: Filter by user ID.
        severity: Filter by severity.
        since: Filter events after this timestamp.

    Returns:
        List[SecurityEvent]: Matching security events.
    """
    redis_client = get_redis()

    try:
        # Get all events (we'll filter in memory since Redis lists don't support query)
        # For production with high volume, consider using Redis Streams or a separate DB
        events_json = await redis_client.lrange(
            SECURITY_LOG_KEY, offset, offset + limit * 10 - 1
        )

        events = []
        for event_json in events_json:
            try:
                data = json.loads(event_json)
                # Apply filters
                if event_type and data.get("event_type") != event_type.value:
                    continue
                if user_id and data.get("user_id") != user_id:
                    continue
                if severity and data.get("severity") != severity.value:
                    continue
                if since:
                    event_time = datetime.fromisoformat(data["timestamp"])
                    if event_time < since:
                        continue

                events.append(SecurityEvent(
                    event_type=SecurityEventType(data["event_type"]),
                    user_id=data.get("user_id"),
                    ip_address=data.get("ip_address"),
                    user_agent=data.get("user_agent"),
                    severity=SecurityEventSeverity(data.get("severity", "info")),
                    message=data.get("message", ""),
                    metadata=data.get("metadata", {}),
                ))

                if len(events) >= limit:
                    break
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error("Failed to parse security event", error=str(e))
                continue

        return events
    except Exception as e:
        logger.error("Failed to retrieve security events", error=str(e))
        return []


async def get_security_event_count(
    *,
    event_type: Optional[SecurityEventType] = None,
    user_id: Optional[str] = None,
    severity: Optional[SecurityEventSeverity] = None,
    since: Optional[datetime] = None,
) -> int:
    """Get count of security events matching filters."""
    # For accurate count, we'd need to scan all events
    # This is a simplified version
    events = await get_security_events(
        limit=10000,
        event_type=event_type,
        user_id=user_id,
        severity=severity,
        since=since,
    )
    return len(events)


# Convenience functions for common events

async def log_login_success(
    *,
    user_id: str,
    email: str,
    ip: str,
    user_agent: str,
    correlation_id: Optional[str] = None,
) -> None:
    """Log successful login."""
    await log_security_event(_build_security_event(
        event_type=SecurityEventType.LOGIN_SUCCESS,
        user_id=user_id,
        email=email,
        ip_address=ip,
        user_agent=user_agent,
        severity=SecurityEventSeverity.INFO,
        message=f"User {email} logged in successfully",
        correlation_id=correlation_id,
    ))


async def log_login_failure(
    *,
    email: str,
    ip: str,
    user_agent: str,
    reason: str,
    correlation_id: Optional[str] = None,
) -> None:
    """Log failed login attempt."""
    await log_security_event(_build_security_event(
        event_type=SecurityEventType.LOGIN_FAILURE,
        email=email,
        ip_address=ip,
        user_agent=user_agent,
        severity=SecurityEventSeverity.WARNING,
        message=f"Failed login attempt for {email}: {reason}",
        metadata={"reason": reason},
        correlation_id=correlation_id,
    ))


async def log_logout(
    *,
    user_id: str,
    email: str,
    ip: str,
    correlation_id: Optional[str] = None,
) -> None:
    """Log user logout."""
    await log_security_event(_build_security_event(
        event_type=SecurityEventType.LOGOUT,
        user_id=user_id,
        email=email,
        ip_address=ip,
        severity=SecurityEventSeverity.INFO,
        message=f"User {email} logged out",
        correlation_id=correlation_id,
    ))


async def log_password_change(
    *,
    user_id: str,
    email: str,
    ip: str,
    correlation_id: Optional[str] = None,
) -> None:
    """Log password change."""
    await log_security_event(_build_security_event(
        event_type=SecurityEventType.PASSWORD_CHANGE,
        user_id=user_id,
        email=email,
        ip_address=ip,
        severity=SecurityEventSeverity.WARNING,
        message=f"Password changed for {email}",
        correlation_id=correlation_id,
    ))


async def log_account_locked(
    *,
    user_id: str,
    email: str,
    ip: str,
    reason: str,
    correlation_id: Optional[str] = None,
) -> None:
    """Log account lockout."""
    await log_security_event(_build_security_event(
        event_type=SecurityEventType.ACCOUNT_LOCKED,
        user_id=user_id,
        email=email,
        ip_address=ip,
        severity=SecurityEventSeverity.CRITICAL,
        message=f"Account locked for {email}: {reason}",
        metadata={"reason": reason},
        correlation_id=correlation_id,
    ))


async def log_session_revoked(
    *,
    user_id: str,
    email: str,
    session_id: str,
    ip: str,
    revoked_by_user: bool = True,
    correlation_id: Optional[str] = None,
) -> None:
    """Log session revocation."""
    await log_security_event(_build_security_event(
        event_type=SecurityEventType.SESSION_REVOKED,
        user_id=user_id,
        email=email,
        ip_address=ip,
        severity=SecurityEventSeverity.INFO,
        message=f"Session {session_id[:8]} revoked for {email}",
        metadata={"session_id": session_id, "revoked_by_user": revoked_by_user},
        correlation_id=correlation_id,
    ))


async def log_all_sessions_revoked(
    *,
    user_id: str,
    email: str,
    count: int,
    ip: str,
    correlation_id: Optional[str] = None,
) -> None:
    """Log all sessions revoked."""
    await log_security_event(_build_security_event(
        event_type=SecurityEventType.ALL_SESSIONS_REVOKED,
        user_id=user_id,
        email=email,
        ip_address=ip,
        severity=SecurityEventSeverity.WARNING,
        message=f"All {count} sessions revoked for {email}",
        metadata={"session_count": count},
        correlation_id=correlation_id,
    ))
