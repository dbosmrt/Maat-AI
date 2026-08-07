"""
Common utilities shared across all microservices.

Provides configuration, exceptions, logging, and utilities.
"""

from server.common.config import get_settings, settings
from server.common.exceptions import (
    MaatException,
    ConfigurationError,
    DatabaseError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    ExternalServiceError,
    AgentError,
)
from server.common.logging import (
    configure_logging,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    LogContext,
    log_function_call,
)

__all__ = [
    # Config
    "get_settings",
    "settings",
    # Exceptions
    "MaatException",
    "ConfigurationError",
    "DatabaseError",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "ExternalServiceError",
    "AgentError",
    # Logging
    "configure_logging",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "LogContext",
    "log_function_call",
]
