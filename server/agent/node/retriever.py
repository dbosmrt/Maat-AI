"""
Retriever Node Module for Legal RAG Chatbot.

Performs a hybrid dense + sparse (BM25) retrieval against the Pinecone
index and fuses the two ranked lists via Reciprocal Rank Fusion (RRF).

The dense path uses the NVIDIA Nemotron embedding model through LangChain's
Pinecone vector store. The BM25 (sparse) path is rebuilt in memory from the
chunks currently stored in Pinecone and cached for the process lifetime;
ingestion must call `invalidate_bm25_cache()` after upserts.
"""

import os
import time
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from agent.utils.logger import get_logger
from agent.model import EmbeddingModels
from agent.state import AgentState
from agent.utils.embedding_utils import VectorDatabases

logger = get_logger(__name__)

# Module-level BM25 cache. Rebuilt lazily on first retrieval after invalidation.
_bm25_cache: Dict[str, BM25Retriever | None] = {"instance": None}

_BM25_TOP_K = 20
_DENSE_TOP_K = 20
_PINECONE_FETCH_TOP_K = 10_000  # serverless cap per query
_RRF_K = 60


def reciprocal_rank_fusion(
    results_lists: List[List[Document]], k: int = _RRF_K
) -> List[Document]:
    """Fuse multiple ranked lists of Documents into a single list via RRF."""
    fused_scores: Dict[str, Dict[str, Any]] = {}
    for docs in results_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            if key not in fused_scores:
                fused_scores[key] = {"doc": doc, "score": 0.0}
            fused_scores[key]["score"] += 1.0 / (rank + k)

    sorted_items = sorted(
        fused_scores.values(), key=lambda x: x["score"], reverse=True
    )
    return [item["doc"] for item in sorted_items]


def invalidate_bm25_cache() -> None:
    """Clear the cached BM25 retriever so it is rebuilt on next query."""
    _bm25_cache["instance"] = None
    logger.info("BM25 cache invalidated. Will rebuild on next retrieval.")


def _fetch_all_chunks_from_pinecone() -> List[Document]:
    """
    Pull every chunk currently stored in the Pinecone index and return them
    as LangChain `Document` objects.

    Uses a zero-vector query with the maximum allowed `top_k`. This is fine
    for serverless indices up to ~10k records — sufficient for the legal
    corpus. For larger corpora, paginate or switch to a dedicated sparse
    index.
    """
    raw_index = VectorDatabases.get_raw_index()
    dimension = int(os.environ.get("PINECONE_DIM", "1024"))

    try:
        result = raw_index.query(
            vector=[0.0] * dimension,
            top_k=_PINECONE_FETCH_TOP_K,
            include_metadata=True,
        )
    except (RuntimeError, ConnectionError, ValueError) as exc:
        logger.error("Failed to fetch chunks from Pinecone for BM25: %s", exc)
        raise

    docs: List[Document] = []
    for match in result.matches:
        metadata = dict(match.metadata or {})
        # langchain-pinecone stores `page_content` under the metadata key
        # `text` by default; fall back to `page_content` for safety.
        page_content = metadata.pop("text", metadata.pop("page_content", ""))
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def _get_bm25_retriever() -> BM25Retriever:
    """Return the cached BM25 retriever, building it from Pinecone on demand."""
    cached = _bm25_cache["instance"]
    if cached is None:
        logger.info("Initializing BM25 keyword index from Pinecone (one-time)…")
        lc_docs = _fetch_all_chunks_from_pinecone()
        if not lc_docs:
            logger.warning("Pinecone index is empty — BM25 will return no hits.")
            lc_docs = [Document(page_content="", metadata={})]
        retriever: BM25Retriever = BM25Retriever.from_documents(lc_docs)
        retriever.k = _BM25_TOP_K
        _bm25_cache["instance"] = retriever
        logger.info("BM25 index initialized (%d chunks).", len(lc_docs))
    result = _bm25_cache["instance"]
    assert result is not None
    return result


def _hybrid_retrieve(hybrid_query: str, vectorstore) -> List[str]:
    """Run dense + BM25 retrieval and return the RRF-fused top-K strings."""
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": _DENSE_TOP_K})
    bm25_retriever = _get_bm25_retriever()

    dense_docs = dense_retriever.invoke(hybrid_query)
    sparse_docs = bm25_retriever.invoke(hybrid_query)

    fused = reciprocal_rank_fusion([dense_docs, sparse_docs])
    return _format_docs(fused[:_DENSE_TOP_K])["documents"]


def retriever_node(state: AgentState) -> dict:
    """
    Retrieve the top-K most relevant document chunks from Pinecone.

    Uses semantic (dense) search with the NVIDIA embedding model and pairs
    it with a BM25 (sparse) pass fused via RRF. On retry iterations the
    rewriter's raw query is used because `decomposed_query` is stale.
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
            vectorstore = VectorDatabases.get_vector_store()
            logger.info(
                "Successfully retrieved chunks (RRF, primary, attempt %d).",
                attempt + 1,
            )
            return {"documents": _hybrid_retrieve(hybrid_query, vectorstore)}
        except (RuntimeError, ConnectionError, ValueError) as exc:
            delay = base_delay * (2 ** attempt)
            if attempt < max_retries - 1:
                logger.warning(
                    "Primary embed failed (attempt %d/%d): %s. Retrying in %ds…",
                    attempt + 1, max_retries, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Primary embedding model exhausted all %d retries.", max_retries
                )

    logger.info("Switching to fallback embedding model…")
    try:
        fallback_embeddings = EmbeddingModels.get_embed_with_fallback()
        vectorstore = VectorDatabases.get_vector_store(embeddings=fallback_embeddings)
        return {"documents": _hybrid_retrieve(hybrid_query, vectorstore)}
    except (RuntimeError, ConnectionError, ValueError) as exc:
        logger.error("Fallback embedding model also failed: %s", exc)
        return {"documents": []}


def _format_docs(docs: List[Document]) -> dict:
    """Format retrieved Documents into prefixed strings for downstream nodes."""
    formatted_docs: List[str] = []
    for i, doc in enumerate(docs):
        context_path = doc.metadata.get("context_path", "Unknown Source")
        formatted_text = f"[Source: {context_path}]\n{doc.page_content}"
        formatted_docs.append(formatted_text)
        logger.debug("Retrieved chunk %d from %s", i + 1, context_path)

    return {"documents": formatted_docs}
