"""Vector database utilities for the Legal RAG Chatbot."""

import os
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings

from agent.model import EmbeddingModels

# Defaults are aligned with `nvidia/nv-embedqa-e5-v5` (NVIDIA Nemotron Embed)
# which emits 1024-dim cosine-friendly normalized vectors.
_DEFAULT_INDEX = "legal-rag"
_DEFAULT_CLOUD = "aws"
_DEFAULT_REGION = "us-east-1"
_DEFAULT_DIM = 1024
_DEFAULT_METRIC = "cosine"

# Expected embedding dimension - must match the Pinecone index
EXPECTED_EMBEDDING_DIM = int(os.environ.get("PINECONE_DIM", _DEFAULT_DIM))


def _resolve_api_key() -> str:
    """Return the Pinecone API key from env, supporting legacy var name."""
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        api_key = os.environ.get("PINECONE_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. Add it to .env or export it before "
            "starting the server."
        )
    return api_key


def _resolve_index_name() -> str:
    """Return the configured Pinecone index name with empty-string fallback."""
    return os.environ.get("PINECONE_INDEX", _DEFAULT_INDEX).strip() or _DEFAULT_INDEX


def _ensure_pinecone_index(pc: Pinecone, name: str) -> None:
    """Create the index if it does not exist. Safe to call at startup."""
    try:
        if pc.has_index(name):
            return
        pc.create_index(
            name=name,
            dimension=int(os.environ.get("PINECONE_DIM", _DEFAULT_DIM)),
            metric=os.environ.get("PINECONE_METRIC", _DEFAULT_METRIC),
            spec=ServerlessSpec(
                cloud=os.environ.get("PINECONE_CLOUD", _DEFAULT_CLOUD),
                region=os.environ.get("PINECONE_REGION", _DEFAULT_REGION),
            ),
        )
    except Exception as exc:
        # If the index was created concurrently or already exists, swallow.
        if "ALREADY_EXISTS" in str(exc) or "already exists" in str(exc).lower():
            return
        raise


def _validate_embedding_dimension(embeddings: Embeddings) -> None:
    """
    Validate that the embedding model produces vectors matching Pinecone index dimension.

    Raises ValueError if dimensions don't match.
    """
    test_vector = embeddings.embed_query("dimension check")
    actual_dim = len(test_vector)
    if actual_dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: model produces {actual_dim}-dim vectors "
            f"but Pinecone index expects {EXPECTED_EMBEDDING_DIM}. "
            f"Set PINECONE_DIM={actual_dim} or use a compatible model."
        )


class VectorDatabases:
    """Single-source vector-store facade backed by Pinecone."""

    @staticmethod
    def get_pinecone_client() -> Pinecone:
        """Return the underlying Pinecone (DB) client."""
        return Pinecone(api_key=_resolve_api_key())

    @staticmethod
    def get_pinecone_index_name() -> str:
        """Return the configured Pinecone index name."""
        return _resolve_index_name()

    @staticmethod
    def get_raw_index():
        """Return the raw `pinecone.Index` handle (low-level API surface)."""
        pc = VectorDatabases.get_pinecone_client()
        name = _resolve_index_name()
        _ensure_pinecone_index(pc, name)
        return pc.Index(name)

    @staticmethod
    def get_vector_store(embeddings: Optional[Embeddings] = None) -> PineconeVectorStore:
        """
        Initialize and return the Pinecone vector store.

        Optionally accepts a pre-built embeddings instance (used by the
        fallback model path in `retriever_node`).
        """
        if embeddings is None:
            embeddings = EmbeddingModels.get_nemotron_embed()
        else:
            # Validate fallback embedding dimension matches index
            _validate_embedding_dimension(embeddings)

        pc = VectorDatabases.get_pinecone_client()
        name = _resolve_index_name()
        _ensure_pinecone_index(pc, name)

        return PineconeVectorStore(
            index_name=name,
            embedding=embeddings,
            pinecone_api_key=_resolve_api_key(),
        )


# Module-level convenience functions (kept for backward compatibility with
# any external callers that may still import the old names).
def get_vector_store(embeddings: Optional[Embeddings] = None) -> PineconeVectorStore:
    """Backward-compatible alias for `VectorDatabases.get_vector_store`."""
    return VectorDatabases.get_vector_store(embeddings)


def get_pinecone_db(embeddings: Optional[Embeddings] = None) -> PineconeVectorStore:
    """Backward-compatible alias for `VectorDatabases.get_vector_store`."""
    return VectorDatabases.get_vector_store(embeddings)
