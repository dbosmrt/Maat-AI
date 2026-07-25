"""Document Ingestion Node for Legal RAG Chatbot.

This module orchestrates the PDF ingestion process by utilizing functions
from `ingestion_utils.py` and also supports direct markdown file ingestion.
"""

from pathlib import Path

from agent.state import AgentState
from agent.utils.ingestion_utils import (
    ingest_documents_to_pinecone,
    process_markdown_files,
    process_pdf_with_fallback,
    validate_paths,
)
from agent.utils.logger import get_logger

logger = get_logger(__name__)


def ingestion_node(state: AgentState) -> dict:
    """
    LangGraph node to handle document ingestion (PDF to Markdown or Markdown to Pinecone).

    Supports two modes:
    1. PDF Ingestion: Converts PDFs in `ingest_input_dir` to Markdown in `ingest_output_dir`
    2. Markdown Ingestion: Loads .md files from `ingest_markdown_dir`, chunks them,
       and embeds them directly into Pinecone

    Args:
        state: Current LangGraph state. For PDF mode: must contain `ingest_input_dir`
            and `ingest_output_dir`. For Markdown mode: must contain `ingest_markdown_dir`.

    Returns:
        A dict with the new `ingest_status` describing the outcome.
    """
    # Check for Markdown ingestion mode
    markdown_dir = state.get("ingest_markdown_dir", "")
    if markdown_dir:
        logger.info("Markdown ingestion mode detected. Input: %s", markdown_dir)
        documents = process_markdown_files(markdown_dir)

        if not documents:
            return {"ingest_status": "Failed: No markdown files found or processed"}

        success = ingest_documents_to_pinecone(documents)

        if success:
            status = f"Completed: Successfully ingested {len(documents)} chunks into Pinecone."
        else:
            status = "Failed: Error during Pinecone ingestion."

        logger.info(status)
        return {"ingest_status": status, "documents": []}

    # Original PDF to Markdown mode
    input_dir = state.get("ingest_input_dir", "")
    output_dir = state.get("ingest_output_dir", "")

    if not input_dir or not output_dir:
        logger.error(
            "ingestion_node failed: ingest_input_dir or ingest_output_dir "
            "not found in state."
        )
        return {"ingest_status": "Failed: Missing directories in state"}

    logger.info(
        "ingestion_node started (PDF mode). Input: %s, Output: %s", input_dir, output_dir
    )

    if not validate_paths(input_dir, output_dir):
        return {"ingest_status": "Failed: Invalid input directory"}

    pdf_files = list(Path(input_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", input_dir)
        return {"ingest_status": "Completed: No PDFs found"}

    logger.info("Found %d PDF(s). Starting ingestion...", len(pdf_files))
    success_count = 0

    for pdf_file in pdf_files:
        if process_pdf_with_fallback(str(pdf_file), output_dir):
            success_count += 1

    status = (
        f"Completed: Successfully converted {success_count}/"
        f"{len(pdf_files)} files."
    )
    logger.info(status)
    return {"ingest_status": status}
