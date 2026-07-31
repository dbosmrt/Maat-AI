"""Custom exception hierarchy for Ma'at Legal AI.

Provides structured exception types for better error handling,
debugging, and user-facing error messages.
"""

from typing import Optional, Any


class MaatError(Exception):
    """Base exception for all Ma'at Legal AI errors."""

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            details: Optional structured details for logging/debugging.
            cause: The original exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(MaatError):
    """Raised when configuration is invalid or missing."""


class LLMError(MaatError):
    """Raised when LLM API calls fail."""

    def __init__(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "LLM rate limit exceeded",
        *,
        retry_after: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.retry_after = retry_after


class LLMOutputParsingError(LLMError):
    """Raised when LLM output cannot be parsed as expected."""

    def __init__(
        self,
        message: str = "Failed to parse LLM output",
        *,
        raw_output: Optional[str] = None,
        expected_format: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.raw_output = raw_output
        self.expected_format = expected_format


class VectorStoreError(MaatError):
    """Raised when vector store operations fail."""

    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        index_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.operation = operation
        self.index_name = index_name


class VectorStoreConnectionError(VectorStoreError):
    """Raised when vector store connection fails."""


class VectorStoreDimensionMismatchError(VectorStoreError):
    """Raised when embedding dimensions don't match index."""

    def __init__(
        self,
        expected_dim: int,
        actual_dim: int,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        message = f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
        super().__init__(message, details=details)
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim


class RetrievalError(MaatError):
    """Raised when document retrieval fails."""

    def __init__(
        self,
        message: str,
        *,
        query: Optional[str] = None,
        retrieval_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.query = query
        self.retrieval_type = retrieval_type


class RetrievalEmptyError(RetrievalError):
    """Raised when retrieval returns no results."""

    def __init__(
        self,
        message: str = "No documents retrieved",
        *,
        query: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, query=query, details=details)


class BM25CacheError(MaatError):
    """Raised when BM25 cache operations fail."""

    def __init__(
        self,
        message: str,
        *,
        cache_path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.cache_path = cache_path
        self.operation = operation


class IngestionError(MaatError):
    """Raised when document ingestion fails."""

    def __init__(
        self,
        message: str,
        *,
        file_path: Optional[str] = None,
        stage: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.file_path = file_path
        self.stage = stage


class PDFParseError(IngestionError):
    """Raised when PDF parsing fails with all available parsers."""

    def __init__(
        self,
        message: str = "All PDF parsers failed",
        *,
        file_path: str,
        parser_errors: Optional[dict[str, str]] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            file_path=file_path,
            stage="parsing",
            details={**(details or {}), "parser_errors": parser_errors or {}},
        )
        self.parser_errors = parser_errors or {}


class ChunkingError(IngestionError):
    """Raised when document chunking fails."""


class EmbeddingError(IngestionError):
    """Raised when document embedding fails."""

    pass


class NodeError(MaatError):
    """Base class for LangGraph node execution errors."""

    def __init__(
        self,
        message: str,
        *,
        node_name: str,
        state_snapshot: Optional[dict[str, Any]] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.node_name = node_name
        self.state_snapshot = state_snapshot or {}


class QualifierError(NodeError):
    """Raised when query qualification fails."""

    pass


class DecomposerError(NodeError):
    """Raised when query decomposition fails."""

    pass


class RetrieverNodeError(NodeError):
    """Raised when retriever node fails."""

    pass


class RerankerError(NodeError):
    """Raised when reranker node fails."""

    pass


class GraderError(NodeError):
    """Raised when grader node fails."""

    pass


class RewriterError(NodeError):
    """Raised when query rewriter fails."""

    pass


class WebSearchError(NodeError):
    """Raised when web search node fails."""

    pass


class GeneratorError(NodeError):
    """Raised when generator node fails."""

    pass


class SessionError(MaatError):
    """Raised when chat session operations fail."""

    def __init__(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.session_id = session_id
        self.user_id = user_id


class AuthenticationError(MaatError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(MaatError):
    """Raised when authorization fails."""

    pass


class ValidationError(MaatError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)
        self.field = field
        self.value = value


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_CODES: dict[type[MaatError], int] = {
    ConfigurationError: 500,
    LLMError: 502,
    LLMTimeoutError: 504,
    LLMRateLimitError: 429,
    LLMOutputParsingError: 502,
    VectorStoreError: 503,
    VectorStoreConnectionError: 503,
    VectorStoreDimensionMismatchError: 500,
    RetrievalError: 503,
    RetrievalEmptyError: 404,
    BM25CacheError: 500,
    IngestionError: 500,
    PDFParseError: 422,
    ChunkingError: 500,
    EmbeddingError: 500,
    NodeError: 500,
    QualifierError: 500,
    DecomposerError: 500,
    RetrieverNodeError: 500,
    RerankerError: 500,
    GraderError: 500,
    RewriterError: 500,
    WebSearchError: 502,
    GeneratorError: 500,
    SessionError: 404,
    AuthenticationError: 401,
    AuthorizationError: 403,
    ValidationError: 422,
}


def get_http_status_code(exception: Exception) -> int:
    """Get HTTP status code for an exception.

    Args:
        exception: The exception to map.

    Returns:
        HTTP status code, defaults to 500 for unknown exceptions.
    """
    for exc_type, code in EXCEPTION_STATUS_CODES.items():
        if isinstance(exception, exc_type):
            return code
    return 500
