"""Configuration classes for Ma'at Legal AI.

Centralized configuration management using Pydantic models for validation
and type safety. Supports environment variable loading with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NVIDIASettings(BaseSettings):
    """NVIDIA NIM API configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NVIDIA_",
        extra="ignore",
    )

    api_key: str = Field(..., description="NVIDIA NIM API key")
    chat_model: str = Field(
        default="nemotron3super",
        description="Default chat model identifier",
    )
    embedding_model: str = Field(
        default="nv_embedqa_e5_v5",
        description="Default embedding model identifier",
    )
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=8192, ge=1, le=32768)

    def get_chat_model_name(self, model_key: str) -> str:
        """Resolve model key to full model name."""
        models = {
            "nemotron3super": "nvidia/llama-3.3-nemotron-super-49b-v1",
            "glm5_2": "z-ai/glm-5.2",
            "sarvam_m": "sarvamai/sarvam-m",
            "minimax_m3": "minimaxai/minimax-m3",
            "nemotron_ultra": "nvidia/nemotron-3-ultra",
            "llama3_1_70b": "meta/llama-3.1-70b-instruct",
        }
        return models.get(model_key, model_key)

    def get_embedding_model_name(self, model_key: str) -> str:
        """Resolve embedding model key to full model name."""
        models = {
            "nv_embedqa_e5_v5": "nvidia/nv-embedqa-e5-v5",
            "nv_embed_v2": "nvidia/nv-embed-v2",
            "bge_m3": "baai/bge-m3",
        }
        return models.get(model_key, model_key)


class PineconeSettings(BaseSettings):
    """Pinecone vector database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PINECONE_",
        extra="ignore",
    )

    api_key: str = Field(..., description="Pinecone API key")
    index_name: str = Field(default="legal-rag", description="Index name")
    cloud: str = Field(default="aws", description="Cloud provider")
    region: str = Field(default="us-east-1", description="Cloud region")
    dimension: int = Field(default=1024, description="Embedding dimension")
    metric: str = Field(default="cosine", description="Distance metric")

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        if v not in {"cosine", "euclidean", "dotproduct"}:
            raise ValueError("metric must be cosine, euclidean, or dotproduct")
        return v


class RetrieverSettings(BaseSettings):
    """Retriever and hybrid search configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVER_",
        extra="ignore",
    )

    dense_top_k: int = Field(default=20, ge=1, le=100)
    bm25_top_k: int = Field(default=20, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=200)
    pinecone_fetch_top_k: int = Field(default=10000, ge=100, le=100000)
    cache_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "vector_store" / "bm25_cache",
        description="BM25 cache directory",
    )


class GraderSettings(BaseSettings):
    """Document grader configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GRADER_",
        extra="ignore",
    )

    min_relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    max_retries_base: int = Field(default=1, ge=0, le=10)
    max_retries_cap: int = Field(default=4, ge=1, le=10)


class WebSearchSettings(BaseSettings):
    """Web search (DuckDuckGo) configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DDGS_",
        extra="ignore",
    )

    region: str = Field(default="in-en", description="DuckDuckGo region")
    max_results: int = Field(default=3, ge=1, le=10)
    query_summarize_threshold: int = Field(default=120, ge=50, le=500)


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        extra="ignore",
    )

    environment: Literal["dev", "server"] = Field(
        default="dev",
        description="Logging mode: dev (console) or server (JSON files)",
    )
    log_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "logs",
        description="Log directory for server mode",
    )
    max_bytes: int = Field(default=10_000_000, description="Max log file size")
    backup_count: int = Field(default=10, description="Number of backup files")


class IngestionSettings(BaseSettings):
    """Document ingestion configuration."""

    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        extra="ignore",
    )

    chunk_size: int = Field(default=2500, ge=100, le=10000)
    chunk_overlap: int = Field(default=400, ge=0, le=2000)
    batch_size: int = Field(default=16, ge=1, le=100)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: int = Field(default=10, ge=1, le=60)
    rate_limit_sleep_seconds: int = Field(default=2, ge=0, le=30)


class ChatHistorySettings(BaseSettings):
    """Chat history and session configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CHAT_",
        extra="ignore",
    )

    max_history_tokens: int = Field(default=4000, ge=500, le=32000)
    encoding: str = Field(default="cl100k_base", description="Tiktoken encoding")


class AppSettings(BaseSettings):
    """Main application settings aggregating all sub-configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurations (loaded from their respective env prefixes)
    nvidia: NVIDIASettings = Field(default_factory=NVIDIASettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    retriever: RetrieverSettings = Field(default_factory=RetrieverSettings)
    grader: GraderSettings = Field(default_factory=GraderSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    chat_history: ChatHistorySettings = Field(default_factory=ChatHistorySettings)

    # App-level settings
    app_name: str = Field(default="Ma'at Legal AI")
    app_version: str = Field(default="1.0.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
    )
    cors_origins: str = Field(default="http://localhost:5173,http://localhost:3000")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


# Global settings instance (lazy loading)
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get the global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings for testing or reconfiguration."""
    global _settings
    _settings = None
