"""
Retriever Node Module for Legal RAG Chatbot.

Performs a hybrid dense + sparse (BM25) retrieval against the Pinecone
index and fuses the two ranked lists via Reciprocal Rank Fusion (RRF).

The dense path uses the NVIDIA Nemotron embedding model through LangChain's
Pinecone vector store. The BM25 (sparse) path is rebuilt in memory from the
chunks currently stored in Pinecone and cached for the process lifetime;
ingestion must call `invalidate_bm25_cache()` after upserts.

BM25 cache is also persisted to disk (pickle) to speed up cold starts.
"""

import os
import time
import threading
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from ..utils.logger import get_logger
from ..model import EmbeddingModels
from ..utils.embedding_utils import get_vector_database_service, get_embedding_models
from .base import BaseNode
from ..state import AgentState

logger = get_logger(__name__)

# Configuration constants
_BM25_TOP_K = 20
_DENSE_TOP_K = 20
_PINECONE_FETCH_TOP_K = 10_000  # serverless cap per query
_RRF_K = 60

# Cache persistence
_BM25_CACHE_DIR = Path(__file__).parent.parent.parent / "vector_store" / "bm25_cache"
_BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_BM25_CACHE_FILE = _BM25_CACHE_DIR / "bm25_retriever.pkl"
_BM25_VERSION_FILE = _BM25_CACHE_DIR / "pinecone_version.txt"


class _EmptyBM25Retriever:
    """Dummy BM25 retriever that always returns empty results when Pinecone is empty."""

    def __init__(self) -> None:
        self.k = _BM25_TOP_K

    def invoke(self, query: str) -> list:
        return []


class RetrieverNode(BaseNode):
    """Node that retrieves relevant document chunks using hybrid search."""

    def __init__(self) -> None:
        super().__init__(name="retriever")
        # Module-level BM25 cache with thread safety
        self._bm25_cache: Dict[str, Optional["BM25Retriever | _EmptyBM25Retriever"]] = {"instance": None}
        self._bm25_lock = threading.Lock()

    def execute(self, state: AgentState) -> dict:
        """Retrieve the top-K most relevant document chunks from Pinecone.

        Uses semantic (dense) search with the NVIDIA embedding model and pairs
        it with a BM25 (sparse) pass fused via RRF. On retry iterations the
        rewriter's raw query is used because `decomposed_query` is stale.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary with "documents" key containing formatted document strings.
        """
        query = state.get("query", "")
        decomposed = state.get("decomposed_query", {}) or {}
        iteration_count = state.get("iteration_count", 0)

        if not query:
            logger.warning("retriever_node: No query found in state.")
            return {"documents": []}

        if iteration_count > 0:
            hybrid_query = query
            logger.info(
                "Retry iteration %d: using rewritten raw query for retrieval.",
                iteration_count,
            )
        else:
            semantic = decomposed.get("semantic_focus", "")
            statutory = decomposed.get("statutory_focus", "")
            procedural = decomposed.get("procedural_focus", "")
            hybrid_query = f"{semantic} {statutory} {procedural}".strip() or query

        logger.info("Retrieving context for hybrid query: '%s'", hybrid_query)

        max_retries = 5
        base_delay = 2

        for attempt in range(max_retries):
            try:
                vectorstore = get_vector_database_service().get_vector_store()
                logger.info(
                    "Successfully retrieved chunks (RRF, primary, attempt %d).",
                    attempt + 1,
                )
                return {"documents": self._hybrid_retrieve(hybrid_query, vectorstore)}
            except (RuntimeError, ConnectionError, ValueError) as exc:
                delay = base_delay * (2 ** attempt)
                if attempt < max_retries - 1:
                    logger.warning(
                        "Primary embed failed (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Primary embedding model exhausted all %d retries.",
                        max_retries,
                    )

        logger.info("Switching to fallback embedding model...")
        try:
            fallback_embeddings = get_embedding_models().get_model_with_fallback()
            vectorstore = get_vector_database_service().get_vector_store(embeddings=fallback_embeddings)
            return {"documents": self._hybrid_retrieve(hybrid_query, vectorstore)}
        except (RuntimeError, ConnectionError, ValueError) as exc:
            logger.error("Fallback embedding model also failed: %s", exc)
            return {"documents": []}

    def _hybrid_retrieve(self, hybrid_query: str, vectorstore) -> List[str]:
        """Run dense + BM25 retrieval and return the RRF-fused top-K strings."""
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": _DENSE_TOP_K})
        bm25_retriever = self._get_bm25_retriever()

        dense_docs = dense_retriever.invoke(hybrid_query)
        sparse_docs = bm25_retriever.invoke(hybrid_query)

        fused = reciprocal_rank_fusion([dense_docs, sparse_docs])
        return self._format_docs(fused[:_DENSE_TOP_K])["documents"]

    def _get_bm25_retriever(self) -> "BM25Retriever | _EmptyBM25Retriever":
        """Return the cached BM25 retriever, building it from Pinecone on demand."""
        # First check without lock (fast path)
        cached = self._bm25_cache["instance"]
        if cached is not None:
            return cached

        # Acquire lock for rebuilding
        with self._bm25_lock:
            # Double-check after acquiring lock
            cached = self._bm25_cache["instance"]
            if cached is not None:
                return cached

            # Try to load from disk cache first
            cached = self._load_bm25_from_cache()
            if cached is not None:
                self._bm25_cache["instance"] = cached
                return cached

            logger.info("Initializing BM25 keyword index from Pinecone (one-time)...")
            lc_docs = self._fetch_all_chunks_from_pinecone()
            if not lc_docs:
                logger.warning("Pinecone index is empty — BM25 will return no hits.")
                empty_retriever = _EmptyBM25Retriever()
                self._bm25_cache["instance"] = empty_retriever
                return empty_retriever

            retriever: BM25Retriever = BM25Retriever.from_documents(lc_docs)
            retriever.k = _BM25_TOP_K
            self._bm25_cache["instance"] = retriever
            # Save to disk for future cold starts
            self._save_bm25_to_cache(retriever)
            logger.info("BM25 index initialized (%d chunks).", len(lc_docs))
            return retriever

    def _fetch_all_chunks_from_pinecone(self) -> List[Document]:
        """Pull every chunk currently stored in the Pinecone index.

        Uses Pinecone's list + fetch approach instead of zero-vector query
        to avoid arbitrary/random results. Fetches in batches to respect API limits.

        Returns:
            List of LangChain Document objects.
        """
        raw_index = get_vector_database_service().get_raw_index()

        docs: List[Document] = []

        try:
            # Get all vector IDs using list (paginated)
            all_ids = []
            paginator = raw_index.list()
            for page in paginator:
                all_ids.extend(page)

            if not all_ids:
                logger.warning("Pinecone index has no vectors — BM25 will be empty.")
                return []

            # Fetch in batches (Pinecone fetch max is 1000 IDs per call)
            batch_size = 1000
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                fetch_result = raw_index.fetch(batch_ids)
                for match in fetch_result.vectors.values():
                    metadata = dict(match.metadata or {})
                    page_content = metadata.pop("text", metadata.pop("page_content", ""))
                    docs.append(Document(page_content=page_content, metadata=metadata))

        except (RuntimeError, ConnectionError, ValueError, AttributeError) as exc:
            logger.error("Failed to fetch chunks from Pinecone for BM25: %s", exc)
            # Fallback: try zero-vector query as last resort (may return random results)
            logger.warning("Falling back to zero-vector query for BM25 rebuild (results may be random)")
            try:
                dimension = int(os.environ.get("PINECONE_DIM", "1024"))
                result = raw_index.query(
                    vector=[0.0] * dimension,
                    top_k=_PINECONE_FETCH_TOP_K,
                    include_metadata=True,
                )
                for match in result.matches:
                    metadata = dict(match.metadata or {})
                    page_content = metadata.pop("text", metadata.pop("page_content", ""))
                    docs.append(Document(page_content=page_content, metadata=metadata))
            except Exception as fallback_exc:
                logger.error("Zero-vector fallback also failed: %s", fallback_exc)
                raise

        return docs

    def _load_bm25_from_cache(self) -> Optional["BM25Retriever | _EmptyBM25Retriever"]:
        """Load BM25 retriever from disk cache if valid."""
        try:
            if not _BM25_CACHE_FILE.exists() or not _BM25_VERSION_FILE.exists():
                return None

            # Check version matches
            with open(_BM25_VERSION_FILE, "r") as f:
                cached_version = f.read().strip()

            current_version = self._get_pinecone_version()
            if cached_version != current_version:
                logger.info("Pinecone index changed (version mismatch), cache invalid.")
                return None

            with open(_BM25_CACHE_FILE, "rb") as f:
                retriever = pickle.load(f)

            logger.info("Loaded BM25 retriever from cache (%d docs).", len(retriever.docs))
            return retriever
        except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
            logger.warning("Failed to load BM25 cache: %s", exc)
            return None

    def _save_bm25_to_cache(self, retriever: BM25Retriever) -> None:
        """Save BM25 retriever to disk cache."""
        try:
            # Save the retriever
            with open(_BM25_CACHE_FILE, "wb") as f:
                pickle.dump(retriever, f)

            # Save version
            current_version = self._get_pinecone_version()
            with open(_BM25_VERSION_FILE, "w") as f:
                f.write(current_version)

            logger.info("Saved BM25 retriever to cache.")
        except Exception as exc:
            logger.warning("Failed to save BM25 cache: %s", exc)

    def _get_pinecone_version(self) -> str:
        """Get a version hash of the Pinecone index contents for cache invalidation.

        Includes namespace-level vector counts and index name to detect content changes
        even when total vector count remains the same (e.g., document updates).
        """
        try:
            raw_index = get_vector_database_service().get_raw_index()
            stats = raw_index.describe_index_stats()
            index_name = get_vector_database_service().get_pinecone_index_name()
            total_vectors = stats.get("total_vector_count", 0)

            # Include namespace-level counts to catch updates within same total count
            namespaces = stats.get("namespaces", {})
            ns_parts = []
            for ns_name, ns_stats in sorted(namespaces.items()):
                ns_count = ns_stats.get("vector_count", 0)
                ns_parts.append(f"{ns_name}:{ns_count}")

            # Hash: index_name + total_vectors + namespace breakdown
            version_string = f"{index_name}:{total_vectors}:{';'.join(ns_parts)}"
            return hashlib.sha256(version_string.encode()).hexdigest()[:16]
        except Exception as exc:
            logger.warning("Could not get Pinecone version for cache: %s", exc)
            return "unknown"

    @staticmethod
    def _format_docs(docs: List[Document]) -> dict:
        """Format retrieved Documents into prefixed strings for downstream nodes."""
        formatted_docs: List[str] = []
        for i, doc in enumerate(docs):
            context_path = doc.metadata.get("context_path", "Unknown Source")
            formatted_text = f"[Source: {context_path}]\n{doc.page_content}"
            formatted_docs.append(formatted_text)
            logger.debug("Retrieved chunk %d from %s", i + 1, context_path)

        return {"documents": formatted_docs}


def reciprocal_rank_fusion(
    results_lists: List[List[Document]],
    k: int = _RRF_K
) -> List[Document]:
    """Fuse multiple ranked lists of Documents into a single list via RRF.

    Args:
        results_lists: List of document lists from different retrievers.
        k: RRF constant (default 60).

    Returns:
        Fused list of documents sorted by combined score.
    """
    fused_scores: Dict[str, Dict[str, Any]] = {}
    for docs in results_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            if key not in fused_scores:
                fused_scores[key] = {"doc": doc, "score": 0.0}
            fused_scores[key]["score"] += 1.0 / (rank + k)

    sorted_items = sorted(
        fused_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )
    return [item["doc"] for item in sorted_items]


# Public API for backward compatibility and ingestion invalidation
_retriever_node = RetrieverNode()


def retriever_node(state: AgentState) -> dict:
    """Backward-compatible function wrapper."""
    return _retriever_node(state)


def invalidate_bm25_cache() -> None:
    """Clear the cached BM25 retriever so it is rebuilt on next query."""
    with _retriever_node._bm25_lock:
        _retriever_node._bm25_cache["instance"] = None
    # Also remove disk cache
    try:
        if _BM25_CACHE_FILE.exists():
            _BM25_CACHE_FILE.unlink()
        if _BM25_VERSION_FILE.exists():
            _BM25_VERSION_FILE.unlink()
    except OSError:
        pass
    logger.info("BM25 cache invalidated (memory and disk). Will rebuild on next retrieval.")

