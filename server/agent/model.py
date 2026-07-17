from langchain_nvidia_ai_endpoints import ChatNVIDIA 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

#load api keys
def _get_nvidia_api_key() -> str:

    key = os.getenv("NVIDIA_NIM_KEY", "")
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add it to .env or set as environment variable."
        )
    return key

class ChatModels:
    @staticmethod
    def get_nemotron3super() -> ChatNVIDIA:
        
        return ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        api_key=_get_nvidia_api_key(),
        temperature=0.6,
        top_p=0.95,
        max_tokens=100000,
    )

    @staticmethod
    def get_glm5_2() -> ChatNVIDIA:
        
        return ChatNVIDIA(
        model="z-ai/glm-5.2",
        api_key=_get_nvidia_api_key(),
        temperature=0.6,
        top_p=0.95,
        max_tokens=100000,
        seed=42
    )

    @staticmethod
    def get_sarvam_m() -> ChatNVIDIA:
        
        return ChatNVIDIA(
        model="sarvamai/sarvam-m",
        api_key=_get_nvidia_api_key(),
        temperature=0.6,
        top_p=0.95,
        max_tokens=100000,
    )

    @staticmethod
    def get_minmax_m3() -> ChatNVIDIA:
        
        return ChatNVIDIA(
        model="minimaxai/minimax-m3",
        api_key=_get_nvidia_api_key(),
        temperature=0.6,
        top_p=0.95,
        max_tokens=100000,
    )



class EmbeddingModels:
    # Ordered list of embedding models to try. All must produce 1024-dim vectors
    # so they remain compatible with the same ChromaDB collection.
    FALLBACK_MODELS = [
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "baai/bge-m3",
    ]

    @staticmethod
    def get_nemotron_embed() -> NVIDIAEmbeddings:
        
        return NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5", 
            api_key=_get_nvidia_api_key(), 
            truncate="END", 
        )

    @staticmethod
    def get_embed_with_fallback() -> NVIDIAEmbeddings:
        """
        Tries each embedding model in FALLBACK_MODELS order.
        Returns the first one that successfully embeds a test string.
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
                logger.info(f"Embedding model '{model_id}' is healthy.")
                return embeddings
            except Exception as e:
                logger.warning(f"Embedding model '{model_id}' failed health check: {e}. Trying next...")
                continue
        
        # If all fallbacks fail, return the primary one anyway and let caller handle errors
        logger.error("All embedding models failed health check. Returning primary as last resort.")
        return NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=_get_nvidia_api_key(),
            truncate="END",
        )
