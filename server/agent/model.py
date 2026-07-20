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


class ChatModels:
    """Static factory methods that produce configured `ChatNVIDIA` models."""

    @staticmethod
    def get_nemotron3super() -> ChatNVIDIA:
        """Return the Nemotron Super 49B chat model."""
        return ChatNVIDIA(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            api_key=_get_nvidia_api_key(),
            temperature=0.6,
            top_p=0.95,
            max_tokens=100000,
        )

    @staticmethod
    def get_glm5_2() -> ChatNVIDIA:
        """Return the GLM 5.2 chat model."""
        return ChatNVIDIA(
            model="z-ai/glm-5.2",
            api_key=_get_nvidia_api_key(),
            temperature=0.6,
            top_p=0.95,
            max_tokens=100000,
            seed=42,
        )

    @staticmethod
    def get_sarvam_m() -> ChatNVIDIA:
        """Return the Sarvam M chat model."""
        return ChatNVIDIA(
            model="sarvamai/sarvam-m",
            api_key=_get_nvidia_api_key(),
            temperature=0.6,
            top_p=0.95,
            max_tokens=100000,
        )

    @staticmethod
    def get_minmax_m3() -> ChatNVIDIA:
        """Return the MiniMax M3 chat model."""
        return ChatNVIDIA(
            model="minimaxai/minimax-m3",
            api_key=_get_nvidia_api_key(),
            temperature=0.6,
            top_p=0.95,
            max_tokens=100000,
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
        Try each embedding model in `FALLBACK_MODELS` in order.

        Returns the first one that embeds a test string without raising.
        If every model fails, returns the primary one so the caller can
        surface the real error.
        """
        from agent.utils.logger import get_logger

        logger = get_logger(__name__)

        for model_id in EmbeddingModels.FALLBACK_MODELS:
            try:
                embeddings = NVIDIAEmbeddings(
                    model=model_id,
                    api_key=_get_nvidia_api_key(),
                    truncate="END",
                )
                # Quick health check: embed a single word
                embeddings.embed_query("test")
                logger.info("Embedding model '%s' is healthy.", model_id)
                return embeddings
            except (RuntimeError, ValueError, ConnectionError) as exc:
                logger.warning(
                    "Embedding model '%s' failed health check: %s. Trying next...",
                    model_id,
                    exc,
                )
                continue

        # If all fallbacks fail, return the primary one anyway
        logger.error(
            "All embedding models failed health check. Returning primary as last resort."
        )
        return NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=_get_nvidia_api_key(),
            truncate="END",
        )
