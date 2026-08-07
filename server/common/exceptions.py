"""
Custom exceptions for Ma'at Legal AI.

Provides structured exception hierarchy for consistent error handling.
"""

from typing import Any, Dict, Optional


class MaatException(Exception):
    """Base exception for Ma'at Legal AI."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


# Configuration Exceptions


class ConfigurationError(MaatException):
    """Configuration error - missing or invalid settings."""

    def __init__(self, message: str = "Configuration error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="CONFIGURATION_ERROR", status_code=500, details=details)


# Authentication & Authorization Exceptions


class AuthenticationError(MaatException):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="AUTHENTICATION_FAILED", status_code=401, details=details)


class InvalidCredentialsError(AuthenticationError):
    """Invalid username/password or token."""

    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(
            message=message,
            details={"code": "INVALID_CREDENTIALS", "status_code": 401}
        )


class TokenExpiredError(AuthenticationError):
    """JWT token has expired."""

    def __init__(self, message: str = "Token has expired"):
        super().__init__(
            message=message,
            details={"code": "TOKEN_EXPIRED", "status_code": 401}
        )


class TokenInvalidError(AuthenticationError):
    """JWT token is invalid or malformed."""

    def __init__(self, message: str = "Invalid token"):
        super().__init__(
            message=message,
            details={"code": "TOKEN_INVALID", "status_code": 401}
        )


class AuthorizationError(MaatException):
    """Authorization failed - insufficient permissions."""

    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="AUTHORIZATION_FAILED", status_code=403, details=details)


# Validation Exceptions


class ValidationError(MaatException):
    """Input validation failed."""

    def __init__(self, message: str = "Validation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="VALIDATION_ERROR", status_code=422, details=details)


class NotFoundError(MaatException):
    """Requested resource not found."""

    def __init__(self, resource: str, identifier: str, details: Optional[Dict[str, Any]] = None):
        message = f"{resource} not found: {identifier}"
        super().__init__(message, code="NOT_FOUND", status_code=404, details=details)


class ConflictError(MaatException):
    """Resource already exists - conflict."""

    def __init__(self, resource: str, identifier: str, details: Optional[Dict[str, Any]] = None):
        message = f"{resource} already exists: {identifier}"
        super().__init__(message, code="CONFLICT", status_code=409, details=details)


# Aliases for backward compatibility
ResourceNotFoundError = NotFoundError
ResourceConflictError = ConflictError


# External Service Exceptions


class ExternalServiceError(MaatException):
    """External service (API, database, etc.) error."""

    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"{service} error: {message}",
            code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details={"service": service, **(details or {})},
        )


class DatabaseError(ExternalServiceError):
    """Database operation failed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("database", message, details)
        self.code = "DATABASE_ERROR"


class VectorStoreError(ExternalServiceError):
    """Vector database (Pinecone) operation failed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("pinecone", message, details)
        self.code = "VECTOR_STORE_ERROR"


class LLMServiceError(ExternalServiceError):
    """LLM API (NVIDIA NIM) error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("nvidia_nim", message, details)
        self.code = "LLM_SERVICE_ERROR"


class RateLimitExceededError(MaatException):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {"retry_after": retry_after} if retry_after else None
        super().__init__(message, code="RATE_LIMIT_EXCEEDED", status_code=429, details=details)


# Backward compatibility alias
RateLimitError = RateLimitExceededError


# Agent Exceptions


class AgentError(MaatException):
    """Agent/LangGraph pipeline error."""

    def __init__(self, message: str = "Agent error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="AGENT_ERROR", status_code=500, details=details)


# Business Logic Exceptions


class ChatSessionNotFoundError(NotFoundError):
    """Chat session not found."""

    def __init__(self, session_id: str):
        super().__init__("ChatSession", session_id)


class ChatMessageNotFoundError(NotFoundError):
    """Chat message not found."""

    def __init__(self, message_id: str):
        super().__init__("ChatMessage", message_id)


class UserNotFoundError(NotFoundError):
    """User not found."""

    def __init__(self, identifier: str):
        super().__init__("User", identifier)


class UserAlreadyExistsError(ConflictError):
    """User already exists."""

    def __init__(self, email: str):
        super().__init__("User", email)


class InsufficientContextError(MaatException):
    """Insufficient context to answer legal question."""

    def __init__(self, message: str = "I do not have enough information to answer this based on the provided documents."):
        super().__init__(message, code="INSUFFICIENT_CONTEXT", status_code=400)


# Exception Mappings for FastAPI

EXCEPTION_STATUS_CODES = {
    AuthenticationError: 401,
    InvalidCredentialsError: 401,
    TokenExpiredError: 401,
    TokenInvalidError: 401,
    AuthorizationError: 403,
    ValidationError: 422,
    ResourceNotFoundError: 404,
    ChatSessionNotFoundError: 404,
    ChatMessageNotFoundError: 404,
    UserNotFoundError: 404,
    ResourceConflictError: 409,
    UserAlreadyExistsError: 409,
    ExternalServiceError: 502,
    DatabaseError: 502,
    VectorStoreError: 502,
    LLMServiceError: 502,
    RateLimitExceededError: 429,
    RateLimitError: 429,
    InsufficientContextError: 400,
    MaatException: 500,
}
