"""Vector database utilities with OOP design."""

import os
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings

from ..model import EmbeddingModels, get_embedding_models
from ..config import get_settings
from ..exceptions import VectorStoreError, VectorStoreDimensionMismatchError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorDatabaseService:
    """Single-source vector-store facade backed by Pinecone."""

    def __init__(
        self,
        embedding_models: Optional[EmbeddingModels] = None,
        index_name: Optional[str] = None,
    ) -> None:
        """Initialize the vector database service.

        Args:
            embedding_models: Embedding model factory (injected for testing).
            index_name: Optional override for Pinecone index name.
        """
        self._settings = get_settings().pinecone
        self._embedding_models = embedding_models or get_embedding_models()
        self._index_name = index_name or self._settings.index_name
        self._pc_client: Optional[Pinecone] = None
        self._index = None

    def _resolve_api_key(self) -> str:
        """Return the Pinecone API key from settings."""
        api_key = self._settings.api_key.strip()
        if not api_key:
            raise VectorStoreError(
                "PINECONE_API_KEY is not set. Add it to .env or export it before starting.",
                operation="get_api_key",
            )
        return api_key

    def _ensure_index(self, pc: Pinecone) -> None:
        """Create the index if it does not exist. Safe to call at startup."""
        try:
            if pc.has_index(self._index_name):
                return
            pc.create_index(
                name=self._index_name,
                dimension=self._settings.dimension,
                metric=self._settings.metric,
                spec=ServerlessSpec(
                    cloud=self._settings.cloud,
                    region=self._settings.region,
                ),
            )
        except Exception as exc:
            # If the index was created concurrently or already exists, swallow.
            if "ALREADY_EXISTS" in str(exc) or "already exists" in str(exc).lower():
                return
            raise VectorStoreError(
                f"Failed to create Pinecone index: {exc}",
                operation="create_index",
                index_name=self._index_name,
                cause=exc,
            ) from exc

    def get_pinecone_client(self) -> Pinecone:
        """Return the underlying Pinecone (DB) client."""
        if self._pc_client is None:
            self._pc_client = Pinecone(api_key=self._resolve_api_key())
        return self._pc_client

    def get_raw_index(self):
        """Return the raw `pinecone.Index` handle (low-level API)."""
        pc = self.get_pinecone_client()
        self._ensure_index(pc)
        if self._index is None:
            self._index = pc.Index(self._index_name)
        return self._index

    def get_vector_store(
        self,
        embeddings: Optional[Embeddings] = None,
    ) -> PineconeVectorStore:
        """
        Initialize and return the Pinecone vector store.

        Optionally accepts a pre-built embeddings instance (used by the
        fallback model path in retriever_node).
        """
        if embeddings is None:
            embeddings = self._embedding_models.get_nemotron_embed()
        else:
            # Validate fallback embedding dimension matches index
            self._validate_embedding_dimension(embeddings)

        pc = self.get_pinecone_client()
        self._ensure_index(pc)

        return PineconeVectorStore(
            index_name=self._index_name,
            embedding=embeddings,
            pinecone_api_key=self._resolve_api_key(),
        )

    def _validate_embedding_dimension(self, embeddings: Embeddings) -> None:
        """
        Validate that the embedding model produces vectors matching Pinecone index dimension.

        Raises:
            VectorStoreDimensionMismatchError: If dimensions don't match.
        """
        test_vector = embeddings.embed_query("dimension check")
        actual_dim = len(test_vector)
        expected_dim = self._settings.dimension

        if actual_dim != expected_dim:
            raise VectorStoreDimensionMismatchError(
                expected_dim=expected_dim,
                actual_dim=actual_dim,
            )

    def get_pinecone_index_name(self) -> str:
        """Return the configured Pinecone index name."""
        return self._index_name


# Backward compatible global instance
_vector_database_service: Optional[VectorDatabaseService] = None


def get_vector_database_service() -> VectorDatabaseService:
    """Get or create the global VectorDatabaseService instance."""
    global _vector_database_service
    if _vector_database_service is None:
        _vector_database_service = VectorDatabaseService()
    return _vector_database_service


# Backward compatible functions
def get_vector_store(embeddings: Optional[Embeddings] = None) -> PineconeVectorStore:
    """Backward-compatible alias for `VectorDatabaseService.get_vector_store`."""
    return get_vector_database_service().get_vector_store(embeddings)


def get_pinecone_db(embeddings: Optional[Embeddings] = None) -> PineconeVectorStore:
    """Backward-compatible alias for `VectorDatabaseService.get_vector_store`."""
    return get_vector_database_service().get_vector_store(embeddings)


def get_raw_index():
    """Backward compatible function."""
    return get_vector_database_service().get_raw_index()


def get_pinecone_index_name() -> str:
    """Backward compatible function."""
    return get_vector_database_service().get_pinecone_index_name()