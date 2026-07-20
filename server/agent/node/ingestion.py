"""Document Ingestion Node for Legal RAG Chatbot.

This module orchestrates the PDF ingestion process by utilizing functions
from `ingestion_utils.py`.
"""

from pathlib import Path

from agent.state import AgentState
from agent.utils.ingestion_utils import (
    parse_pdf_with_docling,
    parse_pdf_with_unstructured,
    validate_paths,
)
from agent.utils.logger import get_logger

logger = get_logger(__name__)


def process_pdf_with_fallback(pdf_path: str, output_dir: str) -> bool:
    """
    Try Docling if GPU is available, otherwise fall back to Unstructured.

    Args:
        pdf_path: Absolute path to the source PDF file.
        output_dir: Directory in which the Markdown output will be saved.

    Returns:
        True on successful conversion; False if both parsers fail.
    """
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    if gpu_available:
        logger.info("GPU detected. Trying high-accuracy Docling parser.")
        if parse_pdf_with_docling(pdf_path, output_dir):
            return True

    logger.info("GPU not detected or Docling failed. Falling back to CPU parsing.")
    return parse_pdf_with_unstructured(pdf_path, output_dir)


def ingestion_node(state: AgentState) -> dict:
    """
    LangGraph node to handle PDF ingestion.

    Reads input/output directories from the state and converts every PDF
    in the input directory to Markdown in the output directory.

    Args:
        state: Current LangGraph state. Must contain `ingest_input_dir`
            and `ingest_output_dir`.

    Returns:
        A dict with the new `ingest_status` describing the outcome.
    """
    input_dir = state.get("ingest_input_dir", "")
    output_dir = state.get("ingest_output_dir", "")

    if not input_dir or not output_dir:
        logger.error(
            "ingestion_node failed: ingest_input_dir or ingest_output_dir "
            "not found in state."
        )
        return {"ingest_status": "Failed: Missing directories in state"}

    logger.info(
        "ingestion_node started. Input: %s, Output: %s", input_dir, output_dir
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
