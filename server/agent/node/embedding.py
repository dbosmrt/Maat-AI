"""
Embedding Node Module for Legal RAG Chatbot.

This module provides the LangGraph node responsible for taking chunked
documents (with metadata), embedding them using the NVIDIA Nemotron
embedding model, and storing them in the Pinecone vector store.
"""

import time
from typing import List

from langchain_core.documents import Document

from agent.state import AgentState
from agent.utils.logger import get_logger
from agent.utils.embedding_utils import VectorDatabases
from agent.node.retriever import invalidate_bm25_cache

logger = get_logger(__name__)


def embedding_node(state: AgentState) -> dict:
    """
    LangGraph node that embeds and persists documents into Pinecone.

    It reads 'documents' from the state, generates embeddings via the
    NVIDIA NIM API, and upserts them in batches to avoid rate-limit errors.
    On success it invalidates the BM25 cache so the next retrieval pass
    rebuilds the sparse layer against the new chunks.
    """
    documents: List[Document] = state.get("documents", [])

    if not documents:
        logger.warning("embedding_node skipped: No documents found in state to embed.")
        return {"ingest_status": "Failed: No documents to embed"}

    logger.info("Initializing Pinecone vector store...")

    try:
        vectorstore = VectorDatabases.get_vector_store()

        logger.info("Adding %d documents to the vector store...", len(documents))

        # Pinecone batches embeddings internally, but the NVIDIA NIM API
        # applies its own per-minute quota. Use a modest batch size that
        # keeps us comfortably under the limit while still being fast.
        batch_size = 16

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Retry transient 429s rather than fail the whole ingest.
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(batch)
                    break
                except (RuntimeError, ConnectionError, ValueError) as batch_error:
                    if "429" in str(batch_error) or attempt < max_retries - 1:
                        logger.warning(
                            "Rate limit on batch %d (attempt %d/%d). Backing off 10s.",
                            i, attempt + 1, max_retries,
                        )
                        time.sleep(10)
                    else:
                        raise

            logger.info("Stored batch %d to %d.", i, i + len(batch))
            time.sleep(2)

        logger.info("Successfully embedded and stored all %d documents.", len(documents))

        # Invalidate the BM25 cache so the next retrieval rebuilds it from
        # the freshly-ingested Pinecone records.
        invalidate_bm25_cache()

        return {
            "ingest_status": "Embedding Completed Successfully",
            "documents": [],
        }

    except (RuntimeError, ConnectionError, ValueError) as exc:
        logger.error("Failed to embed and store documents: %s", exc)
        return {"ingest_status": f"Embedding Failed: {exc}"}
