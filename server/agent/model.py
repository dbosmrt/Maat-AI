"""Model initialization for the Legal RAG Chatbot.

Provides factory classes for chat and embedding models backed by NVIDIA NIM.
"""

import os
from pathlib import Path

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

try:
    from dotenv import load_dotenv

    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

# load API keys


def _get_nvidia_api_key() -> str:
    """Return the NVIDIA NIM API key from the environment."""
    key = os.getenv("NVIDIA_NIM_KEY", "")
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add it to .env or set as environment variable."
        )
    return key


def _get_chat_model_config() -> dict:
    """Return chat model configuration from environment with defaults."""
    return {
        "nemotron3super_model": os.getenv(
            "CHAT_MODEL_NEMOTRON3SUPER", "nvidia/llama-3.3-nemotron-super-49b-v1"
        ),
        "glm5_2_model": os.getenv("CHAT_MODEL_GLM5_2", "z-ai/glm-5.2"),
        "sarvam_m_model": os.getenv("CHAT_MODEL_SARVAM_M", "sarvamai/sarvam-m"),
        "minmax_m3_model": os.getenv("CHAT_MODEL_MINMAX_M3", "minimaxai/minimax-m3"),
        "temperature": float(os.getenv("CHAT_TEMPERATURE", "0.6")),
        "top_p": float(os.getenv("CHAT_TOP_P", "0.95")),
        "max_tokens": int(os.getenv("CHAT_MAX_TOKENS", "8192")),
    }


class ChatModels:
    """Static factory methods that produce configured `ChatNVIDIA` models."""

    @staticmethod
    def get_nemotron3super() -> ChatNVIDIA:
        """Return the Nemotron Super 49B chat model."""
        cfg = _get_chat_model_config()
        return ChatNVIDIA(
            model=cfg["nemotron3super_model"],
            api_key=_get_nvidia_api_key(),
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
        )

    @staticmethod
    def get_glm5_2() -> ChatNVIDIA:
        """Return the GLM 5.2 chat model."""
        cfg = _get_chat_model_config()
        return ChatNVIDIA(
            model=cfg["glm5_2_model"],
            api_key=_get_nvidia_api_key(),
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
            seed=42,
        )

    @staticmethod
    def get_sarvam_m() -> ChatNVIDIA:
        """Return the Sarvam M chat model."""
        cfg = _get_chat_model_config()
        return ChatNVIDIA(
            model=cfg["sarvam_m_model"],
            api_key=_get_nvidia_api_key(),
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
        )

    @staticmethod
    def get_minmax_m3() -> ChatNVIDIA:
        """Return the MiniMax M3 chat model."""
        cfg = _get_chat_model_config()
        return ChatNVIDIA(
            model=cfg["minmax_m3_model"],
            api_key=_get_nvidia_api_key(),
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
        )


class EmbeddingModels:
    """Static factory methods for NVIDIA embedding models."""

    # Ordered list of embedding models to try. All must produce 1024-dim vectors
    # so they remain compatible with the same Pinecone collection.
    FALLBACK_MODELS = [
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "baai/bge-m3",
    ]

    @staticmethod
    def get_nemotron_embed() -> NVIDIAEmbeddings:
        """Return the primary Nemotron embedding model."""
        return NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=_get_nvidia_api_key(),
            truncate="END",
        )

    @staticmethod
    def get_embed_with_fallback() -> NVIDIAEmbeddings:
        """
        Try each embedding model in `FALLBACK_MODELS` in order with exponential backoff.

        Returns the first one that embeds a test string without raising.
        If every model fails, returns the primary one so the caller can
        surface the real error.
        """
        import time
        from agent.utils.logger import get_logger

        logger = get_logger(__name__)

        for model_id in EmbeddingModels.FALLBACK_MODELS:
            # Retry each model with exponential backoff
            max_retries = 3
            base_delay = 1.0
            for attempt in range(max_retries):
                try:
                    embeddings = NVIDIAEmbeddings(
                        model=model_id,
                        api_key=_get_nvidia_api_key(),
                        truncate="END",
                    )
                    # Quick health check: embed a single word
                    embeddings.embed_query("test")
                    if attempt > 0:
                        logger.info("Embedding model '%s' health check succeeded on attempt %d.", model_id, attempt + 1)
                    else:
                        logger.info("Embedding model '%s' is healthy.", model_id)
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

        # If all fallbacks fail, return the primary one anyway
        logger.error(
            "All embedding models failed health check. Returning primary as last resort."
        )
        return NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=_get_nvidia_api_key(),
            truncate="END",
        )
