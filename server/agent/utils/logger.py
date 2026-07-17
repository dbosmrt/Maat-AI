"""
Ma-at Advanced Logging Configuration

Production-grade logging with two operational modes:
  - DEVELOPER mode: Colorful, verbose console output for local debugging.
  - SERVER mode:    Structured JSON file logging with rotation for production/staging.

Set via environment variable:  ENVIRONMENT=dev | server  (default: dev)

Usage:
    from agent.utils.logger import get_logger, log_node_event, log_system_error

    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import logging.handlers
import os
import sys
import json
import traceback as tb_module
from datetime import datetime, timezone
from contextvars import ContextVar
from functools import wraps
import time


# Constants & Configuration


# Calculate project root dynamically: /server/agent/utils/logger.py -> ../../../logs
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")

# Correlation ID for request tracing (set per-request in API layer)
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")

# ANSI Color Codes (for Developer Mode Console Output)

class _Colors:
    """ANSI escape codes for colored terminal output."""
    RESET      = "\033[0m"
    BOLD       = "\033[1m"
    DIM        = "\033[2m"
    ITALIC     = "\033[3m"
    UNDERLINE  = "\033[4m"

    # Foreground colors
    BLACK      = "\033[30m"
    RED        = "\033[31m"
    GREEN      = "\033[32m"
    YELLOW     = "\033[33m"
    BLUE       = "\033[34m"
    MAGENTA    = "\033[35m"
    CYAN       = "\033[36m"
    WHITE      = "\033[37m"

    # Bright foreground
    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    # Background colors
    BG_RED     = "\033[41m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_WHITE   = "\033[47m"

# Level -> (color, label)
_LEVEL_STYLES = {
    logging.DEBUG:    (_Colors.DIM + _Colors.CYAN,        "DEBUG"),
    logging.INFO:     (_Colors.BRIGHT_GREEN,            "INFO "),
    logging.WARNING:  (_Colors.BRIGHT_YELLOW,           "WARN "),
    logging.ERROR:    (_Colors.BRIGHT_RED,              "ERROR"),
    logging.CRITICAL: (_Colors.BOLD + _Colors.BG_RED + _Colors.WHITE, "FATAL"),
}

# Custom Formatters

class ColoredDevFormatter(logging.Formatter):
    """
    Rich, colorful formatter for developer console output.

    Format:
      14:23:05 | INFO  | agent.node.grader:grader_node:72 | Message here
    """

    def format(self, record):
        color, label = _LEVEL_STYLES.get(
            record.levelno, (_Colors.WHITE, record.levelname)
        )

        timestamp = datetime.now().strftime("%H:%M:%S")
        time_str = f"{_Colors.DIM}{timestamp}{_Colors.RESET}"

        level_str = f"{color}{label}{_Colors.RESET}"

        # Shorten the logger name for readability (e.g. agent.node.grader -> grader)
        short_name = record.name.rsplit(".", 1)[-1] if "." in record.name else record.name
        location = (
            f"{_Colors.BRIGHT_BLUE}{short_name}"
            f"{_Colors.DIM}:{record.funcName}:{record.lineno}{_Colors.RESET}"
        )

        message = record.getMessage()

        # Correlation ID (only show if set)
        corr_id = correlation_id.get("-")
        corr_str = ""
        if corr_id != "-":
            corr_str = f" {_Colors.DIM}[{corr_id}]{_Colors.RESET}"

        line = f"{time_str} │ {level_str} │ {location}{corr_str} │ {message}"

        # Append exception info with red color
        if record.exc_info and record.exc_info[0] is not None:
            exc_text = self.formatException(record.exc_info)
            line += f"\n{_Colors.RED}{exc_text}{_Colors.RESET}"

        return line


class JsonServerFormatter(logging.Formatter):
    """
    Structured JSON formatter for production/server log files.
    Each line is a valid JSON object — ideal for ELK, Loki, Datadog, etc.
    """

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get("-"),
        }

        # Add exception details if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add any extra structured fields passed via logger.info("msg", extra={...})
        for key in ("node_id", "status", "tokens_used", "duration_ms", "error_payload"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


# Filters

class CorrelationFilter(logging.Filter):
    """Injects correlation_id into every log record for structured logging."""

    def filter(self, record):
        record.correlation_id = correlation_id.get("-")
        return True


# Log Directory Bootstrap

def _ensure_log_dir():
    """Creates the log directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


# Logger Factory

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Behavior depends on the ENVIRONMENT env var:
      - "dev" (default):  DEBUG level, colored console output
      - "server":         INFO level, JSON file output with rotation

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handler attachment on re-imports
    if logger.hasHandlers():
        return logger

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    env = os.getenv("ENVIRONMENT", "dev").lower()

    if env == "server":
        _configure_server_logger(logger)
    else:
        _configure_dev_logger(logger)

    return logger


def _configure_dev_logger(logger: logging.Logger):
    """
    Developer mode:
      - DEBUG level (everything visible)
      - Colorful console output
      - No file handlers (keep it lightweight)
    """
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ColoredDevFormatter())
    console_handler.addFilter(CorrelationFilter())

    logger.addHandler(console_handler)


def _configure_server_logger(logger: logging.Logger):
    """
    Server/production mode:
      - INFO level for general logs
      - Structured JSON output to rotating file
      - Separate error log for quick triage
      - Console handler at WARNING+ for container stdout
    """
    _ensure_log_dir()
    logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter level

    # --- Handler 1: Main application log (JSON, rotating) ---
    main_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "maat_server.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
        encoding="utf-8",
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(JsonServerFormatter())
    main_handler.addFilter(CorrelationFilter())

    # --- Handler 2: Error-only log for rapid triage ---
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "maat_errors.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JsonServerFormatter())
    error_handler.addFilter(CorrelationFilter())

    # --- Handler 3: Console (container stdout) at WARNING+ ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(JsonServerFormatter())
    console_handler.addFilter(CorrelationFilter())

    logger.addHandler(main_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)


# Structured Node Event Logger

def log_node_event(
    node_name: str,
    status: str,
    tokens_used: int = None,
    duration_ms: float = None,
    error_payload: str = None,
):
    """
    Logs structured operational telemetry for a pipeline node execution.

    In SERVER mode, events are written as JSON to a daily node-events log file.
    In DEV mode, events are also printed to the console logger with rich formatting.

    Args:
        node_name:     Identifier of the node (e.g. "grader_node", "retriever_node")
        status:        Event status (e.g. "SUCCESS", "FAILURE", "RETRY", "PARSING_RETRY")
        tokens_used:   Optional token count consumed by the LLM call.
        duration_ms:   Optional execution time in milliseconds.
        error_payload: Optional error message string for failure events.
    """
    _ensure_log_dir()
    env = os.getenv("ENVIRONMENT", "dev").lower()

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_id": node_name,
        "status": status,
        "correlation_id": correlation_id.get("-"),
    }

    if tokens_used is not None:
        event["tokens_used"] = tokens_used
    if duration_ms is not None:
        event["duration_ms"] = round(duration_ms, 2)
    if error_payload:
        event["error_payload"] = error_payload

    # Always write to the node events log file
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"{date_str}_node_events.log")

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:
        # Last-resort fallback — never crash the pipeline because of logging
        sys.stderr.write(f"[LOGGING FAILURE] Could not write node event: {e}\n")

    # In dev mode, also emit through the regular logger for colored console output
    if env != "server":
        node_logger = get_logger(f"node.{node_name}")
        extra_info = ""
        if tokens_used is not None:
            extra_info += f" | tokens={tokens_used}"
        if duration_ms is not None:
            extra_info += f" | {duration_ms:.0f}ms"
        if error_payload:
            extra_info += f" | err={error_payload[:120]}"

        log_level = logging.ERROR if status == "FAILURE" else logging.INFO
        node_logger.log(log_level, f"[{status}]{extra_info}")


def log_system_error(error_trace: str):
    """
    Logs full stack traces and system-level errors to a centralized error file.
    Used for catastrophic failures that need immediate attention.

    Args:
        error_trace: Full traceback string (from traceback.format_exc()).
    """
    _ensure_log_dir()
    log_file = os.path.join(LOG_DIR, "system_errors.log")
    timestamp = datetime.now(timezone.utc).isoformat()
    corr_id = correlation_id.get("-")

    entry = {
        "timestamp": timestamp,
        "correlation_id": corr_id,
        "error_trace": error_trace,
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        sys.stderr.write(f"[LOGGING FAILURE] Could not write system error: {e}\n")



# Performance Timing Decorator

def log_execution_time(func):
    """
    Decorator that logs the execution time of a function.
    Uses the function's module logger.

    Usage:
        @log_execution_time
        def my_slow_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            func_logger.info(
                f"{func.__qualname__} completed in {elapsed_ms:.1f}ms"
            )
            return result
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            func_logger.error(
                f"{func.__qualname__} FAILED after {elapsed_ms:.1f}ms"
            )
            raise
    return wrapper



# Correlation ID Helpers (for API middleware)

def set_correlation_id(cid: str):
    """Set the correlation ID for the current async context / request."""
    correlation_id.set(cid)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id.get("-")
