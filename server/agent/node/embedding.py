"""
Embedding Node Module for Legal RAG Chatbot.

This module provides the LangGraph node responsible for taking chunked 
documents (with metadata), embedding them using the NVIDIA Nemotron 
embedding model, and storing them in the ChromaDB vector store.
"""

import logging
import os
import time
from typing import List

from langchain_core.documents import Document

from agent.state import AgentState
from agent.utils.embedding_utils import get_vector_store, VECTOR_STORE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def embedding_node(state: AgentState) -> dict:
    """
    LangGraph node to handle Document Embedding and Storage.
    It reads 'documents' from the state, generates embeddings using NVIDIA's API, 
    and persists them to the local ChromaDB.
    """
    documents: List[Document] = state.get("documents", [])
    
    if not documents:
        logger.warning("embedding_node skipped: No documents found in state to embed.")
        return {"ingest_status": "Failed: No documents to embed"}
        
    logger.info("Initializing Chroma vector store at %s...", VECTOR_STORE_DIR)
    
    try:
        vectorstore = get_vector_store()
        
        # Batch add documents to vector store
        logger.info("Adding %d documents to the vector store...", len(documents))
        
        # ChromaDB automatically handles chunking for the DB insertion and embedding API
        # but large batches might hit NVIDIA API rate limits or CUDA OOM, so we add them in smaller batches.
        batch_size = 16
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            
            # Simple retry loop for 429 Too Many Requests
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(batch)
                    break
                except Exception as batch_error:
                    if "429" in str(batch_error) or attempt < max_retries - 1:
                        logger.warning("Hit API rate limit on batch %d. Sleeping for 10 seconds... (Attempt %d/%d)", i, attempt+1, max_retries)
                        time.sleep(10)
                    else:
                        raise batch_error
                        
            logger.info("Embedded and stored batch %d to %d...", i, i + len(batch))
            # Sleep briefly to avoid hitting rate limits too quickly
            time.sleep(2)
            
        logger.info("Successfully embedded and stored all %d documents.", len(documents))
        
        # Clear the documents from state since they are stored to save memory
        return {
            "ingest_status": "Embedding Completed Successfully",
            "documents": []
        }
        
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to embed and store documents: %s", e)
        return {"ingest_status": f"Embedding Failed: {e}"}
