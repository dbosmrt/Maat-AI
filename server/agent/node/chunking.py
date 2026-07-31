"""Markdown Chunking Module for Legal RAG Chatbot.

This module provides a LangGraph node that chunks large Markdown files.
"""

from pathlib import Path

from ..state import AgentState
from ..utils.chunking_utils import chunk_markdown_file
from ..utils.logger import get_logger

logger = get_logger(__name__)


def chunking_node(state: AgentState) -> dict:
    """
    LangGraph node to handle Markdown chunking.

    Reads the output directory from the state (which contains .md files),
    chunks them, and updates the state with the resulting Documents.

    Args:
        state: Current LangGraph state. Must contain `ingest_output_dir`.

    Returns:
        A dict containing the chunked documents and the updated status.
    """
    md_dir = state.get("ingest_output_dir", "")

    if not md_dir:
        logger.error("chunking_node failed: ingest_output_dir not found in state.")
        return {"ingest_status": "Failed: Missing markdown directory in state"}

    md_path = Path(md_dir)
    if not md_path.exists() or not md_path.is_dir():
        logger.error("Markdown directory not found: %s", md_dir)
        return {"ingest_status": "Failed: Invalid markdown directory"}

    md_files = list(md_path.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s", md_dir)
        return {"ingest_status": "Completed (No files to chunk)"}

    all_chunks = []
    for md_file in md_files:
        chunks = chunk_markdown_file(str(md_file))
        all_chunks.extend(chunks)

    logger.info("Total chunks created across all files: %d", len(all_chunks))

    return {
        "documents": all_chunks,
        "ingest_status": "Chunking Completed",
    }

