"""Model initialization and factory for Ma'at Legal AI.

Provides factory classes for chat and embedding models backed by NVIDIA NIM.
Supports dynamic model selection based on user settings with dependency injection.
"""

import os
import time
from typing import Any, Dict, Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from .config import get_settings
from .exceptions import ConfigurationError, LLMError, LLMOutputParsingError
from .utils.logger import get_logger


class ChatModels:
    """Factory for creating NVIDIA NIM chat models with dependency injection."""

    def __init__(self, settings: Optional[Any] = None) -> None:
        """Initialize with configuration.

        Args:
            settings: Configuration object. If None, uses global settings.
        """
        self._settings = settings or get_settings().nvidia

    # Available NVIDIA NIM chat models
    AVAILABLE_CHAT_MODELS = {
        "nemotron3super": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "glm5_2": "z-ai/glm-5.2",
        "sarvam_m": "sarvamai/sarvam-m",
        "minimax_m3": "minimaxai/minimax-m3",
        "nemotron_ultra": "nvidia/nemotron-3-ultra",
        "llama3_1_70b": "meta/llama-3.1-70b-instruct",
        "sarvam": "sarvamai/sarvam-m",  # Alias for sarvam_m
    }

    def get_model(
        self,
        model_id: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatNVIDIA:
        """Create a chat model by model identifier.

        Args:
            model_id: Model identifier (key from AVAILABLE_CHAT_MODELS or full model name)
            temperature: Override default temperature
            top_p: Override default top_p
            max_tokens: Override default max_tokens

        Returns:
            Configured ChatNVIDIA instance

        Raises:
            ConfigurationError: If model configuration is invalid.
            LLMError: If model initialization fails.
        """
        # Resolve model name
        model_name = self.AVAILABLE_CHAT_MODELS.get(model_id, model_id)

        config = {
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "top_p": top_p if top_p is not None else self._settings.top_p,
            "max_tokens": max_tokens if max_tokens is not None else self._settings.max_tokens,
        }

        try:
            return ChatNVIDIA(
                model=model_name,
                api_key=self._settings.api_key,
                **config,
            )
        except Exception as exc:
            raise LLMError(
                f"Failed to initialize chat model '{model_name}': {exc}",
                model=model_name,
                cause=exc,
            ) from exc

    def get_from_user_settings(
        self,
        preferred_model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatNVIDIA:
        """Create a chat model using user preferences.

        Args:
            preferred_model: User's preferred model (key or full name)
            temperature: User's temperature setting
            top_p: User's top_p setting
            max_tokens: User's max_tokens setting

        Returns:
            Configured ChatNVIDIA instance
        """
        # Default to Nemotron 3 Super if no preference
        model_name = preferred_model or "nemotron3super"
        model_name = self.AVAILABLE_CHAT_MODELS.get(model_name, model_name)

        return self.get_model(
            model_id=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def get_sarvam_m(self) -> ChatNVIDIA:
        """Get the Sarvam-M model (primary model for structured output tasks)."""
        return self.get_model("sarvam_m")

    def get_nemotron_3_super(self) -> ChatNVIDIA:
        """Get the Nemotron 3 Super model (primary model for generation)."""
        return self.get_model("nemotron3super")


class EmbeddingModels:
    """Factory for NVIDIA embedding models with fallback support."""

    # Ordered list of embedding models to try (fallback chain)
    FALLBACK_MODELS = [
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "baai/bge-m3",
    ]

    # Available embedding model keys
    AVAILABLE_EMBEDDING_MODELS = {
        "nv_embedqa_e5_v5": "nvidia/nv-embedqa-e5-v5",
        "nv_embed_v2": "nvidia/nv-embed-v2",
        "bge_m3": "baai/bge-m3",
    }

    def __init__(self, settings: Optional[Any] = None) -> None:
        """Initialize with configuration.

        Args:
            settings: Configuration object. If None, uses global settings.
        """
        self._settings = settings or get_settings().nvidia

    def get_model(self, model_id: str = "nv_embedqa_e5_v5") -> NVIDIAEmbeddings:
        """Create an embedding model by model identifier.

        Args:
            model_id: Model identifier (key from AVAILABLE_EMBEDDING_MODELS or full model name)

        Returns:
            Configured NVIDIAEmbeddings instance

        Raises:
            LLMError: If model initialization fails.
        """
        model_name = self.AVAILABLE_EMBEDDING_MODELS.get(model_id, model_id)

        try:
            return NVIDIAEmbeddings(
                model=model_name,
                api_key=self._settings.api_key,
                truncate="END",
            )
        except Exception as exc:
            raise LLMError(
                f"Failed to initialize embedding model '{model_name}': {exc}",
                model=model_name,
                cause=exc,
            ) from exc

    def get_nemotron_embed(self) -> NVIDIAEmbeddings:
        """Get the primary Nemotron embedding model."""
        return self.get_model("nv_embedqa_e5_v5")

    def get_model_with_fallback(self) -> NVIDIAEmbeddings:
        """Try each embedding model in FALLBACK_MODELS with retries.

        Returns the first one that responds successfully.

        Returns:
            Working NVIDIAEmbeddings instance

        Raises:
            LLMError: If all fallback models fail.
        """
        logger = get_logger(__name__)

        for model_id in self.FALLBACK_MODELS:
            max_retries = 3
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    embeddings = NVIDIAEmbeddings(
                        model=model_id,
                        api_key=self._settings.api_key,
                        truncate="END",
                    )
                    # Health check
                    embeddings.embed_query("test")
                    if attempt > 0:
                        logger.info(
                            "Embedding model '%s' health check succeeded on attempt %d",
                            model_id,
                            attempt + 1,
                        )
                    else:
                        logger.info("Embedding model '%s' is healthy", model_id)
                    return embeddings
                except (RuntimeError, ValueError, ConnectionError) as exc:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Embedding model '%s' failed health check (attempt %d/%d): %s. Retrying in %.1fs...",
                            model_id,
                            attempt + 1,
                            max_retries,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "Embedding model '%s' failed health check after %d attempts: %s. Trying next...",
                            model_id,
                            max_retries,
                            exc,
                        )

        # If all fallbacks fail, return primary model
        logger.error("All embedding models failed health check. Returning primary as last resort.")
        return self.get_model("nv_embedqa_e5_v5")


# For backward compatibility - global instances
_default_chat_models: Optional[ChatModels] = None
_default_embedding_models: Optional[EmbeddingModels] = None


def get_chat_models(settings: Optional[Any] = None) -> ChatModels:
    """Get or create the global ChatModels instance."""
    global _default_chat_models
    if _default_chat_models is None or settings is not None:
        _default_chat_models = ChatModels(settings)
    return _default_chat_models


def get_embedding_models(settings: Optional[Any] = None) -> EmbeddingModels:
    """Get or create the global EmbeddingModels instance."""
    global _default_embedding_models
    if _default_embedding_models is None or settings is not None:
        _default_embedding_models = EmbeddingModels(settings)
    return _default_embedding_models


# Backward compatible static methods
def get_nvidia_api_key() -> str:
    """Get NVIDIA API key from settings."""
    return get_settings().nvidia.api_key


def get_chat_model_config() -> Dict[str, Any]:
    """Get default chat model configuration."""
    return get_settings().nvidia.model_dump()
