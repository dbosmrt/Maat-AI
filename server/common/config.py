"""
Application configuration using Pydantic Settings.

Centralized configuration management with environment variable support.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # APPLICATION

    APP_NAME: str = "Ma'at Legal AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")

    # SECURITY

    JWT_SECRET_KEY: str = Field(..., min_length=32, description="JWT signing secret (min 32 chars)")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 1

    # Encryption for sensitive data at rest
    ENCRYPTION_KEY: Optional[str] = Field(default=None, description="Fernet encryption key for sensitive data")

    # DATABASE

    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "maat_ai"


    # REDIS (for rate limiting, sessions, caching)
    REDIS_URI: str = "redis://localhost:6379"


    # CORS

    CORS_ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",")]


    # RATE LIMITING

    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: str = "1minute"

    # Auth endpoints stricter limits
    AUTH_RATE_LIMIT_REQUESTS: int = 5
    AUTH_RATE_LIMIT_WINDOW: str = "1minute"


    # NVIDIA NIM

    NVIDIA_NIM_KEY: str = Field(..., description="NVIDIA NIM API key")

    # Model defaults (can be overridden per-user in settings)
    CHAT_MODEL_NEMOTRON3SUPER: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    CHAT_MODEL_GLM5_2: str = "z-ai/glm-5.2"
    CHAT_MODEL_SARVAM_M: str = "sarvamai/sarvam-m"
    CHAT_MODEL_MINIMAX_M3: str = "minimaxai/minimax-m3"
    CHAT_MODEL_NEMOTRON_ULTRA: str = "nvidia/nemotron-3-ultra"
    CHAT_MODEL_LLAMA3_1_70B: str = "meta/llama-3.1-70b-instruct"
    CHAT_TEMPERATURE: float = 0.6
    CHAT_TOP_P: float = 0.95
    CHAT_MAX_TOKENS: int = 8192


    # PINECONE VECTOR DATABASE

    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_INDEX: str = "legal-rag"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_DIM: int = 1024
    PINECONE_METRIC: str = "cosine"


    # OPTIONAL - WEB SEARCH

    DDGS_REGION: str = "in-en"
    DDGS_MAX_RESULTS: int = 3


    # OPTIONAL - QUERY PROCESSING

    QUERY_SUMMARIZE_THRESHOLD: int = 120
    MAX_HISTORY_TOKENS: int = 4000


    # OPTIONAL - LANGSMITH OBSERVABILITY

    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "maat-legal-ai"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_TRACING: bool = False
    LANGSMITH_TRACING_V2: bool = False


    # LOGGING

    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_FORMAT: str = Field(default="json", pattern="^(json|console)$")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
