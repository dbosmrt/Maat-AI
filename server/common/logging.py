"""
Structured logging configuration for Ma'at Legal AI.

Provides JSON-formatted structured logging with correlation IDs.
"""

import asyncio
import logging
import sys
from contextvars import ContextVar
from typing import Any, Optional

import structlog

from server.common.config import settings

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def add_correlation_id(_logger, _method_name, event_dict) -> dict:  # noqa: ANN001
    """Add correlation ID to log entries if available."""
    cid = correlation_id_var.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def add_service_info(_logger, _method_name, event_dict) -> dict:  # noqa: ANN001
    """Add service metadata to log entries."""
    event_dict["service"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    event_dict["environment"] = settings.ENVIRONMENT
    return event_dict


def configure_logging() -> None:
    """Configure structlog with JSON or console output based on settings."""
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_correlation_id,
        add_service_info,
    ]

    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id_var.get()


class LogContext:
    """Context manager for adding structured context to logs."""

    def __init__(self, **kwargs: Any):
        self.context = kwargs
        self.token = None

    def __enter__(self) -> "LogContext":
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())
        if self.token:
            structlog.contextvars.reset_contextvars(token=self.token)


def log_function_call(logger: structlog.BoundLogger):
    """Decorator to log function entry/exit with timing."""
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.debug(f"Entering {func.__name__}", function=func.__name__, args=args, kwargs=kwargs)
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug(f"Exiting {func.__name__}", function=func.__name__, elapsed_ms=elapsed * 1000)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"Error in {func.__name__}", function=func.__name__, elapsed_ms=elapsed * 1000, error=str(e))
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.debug(f"Entering {func.__name__}", function=func.__name__, args=args, kwargs=kwargs)
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug(f"Exiting {func.__name__}", function=func.__name__, elapsed_ms=elapsed * 1000)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"Error in {func.__name__}", function=func.__name__, elapsed_ms=elapsed * 1000, error=str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Initialize logging on module import
configure_logging()
