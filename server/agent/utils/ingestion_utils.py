"""Helpers for ingesting legal PDF documents and markdown files."""

import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from agent.utils.chunking_utils import chunk_markdown_file
from agent.utils.embedding_utils import VectorDatabases
from agent.utils.logger import get_logger
from agent.node.retriever import invalidate_bm25_cache

logger = get_logger(__name__)


def validate_paths(input_dir: str, output_dir: str) -> bool:
    """Check that the input directory exists and create the output dir as needed.

    Args:
        input_dir: Path to the directory containing source PDFs.
        output_dir: Path to the directory where Markdown will be written.

    Returns:
        True if the input path is a directory; False otherwise.
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        logger.error("Input directory does not exist or is invalid: %s", input_dir)
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return True


def _load_markdown_files(markdown_dir: str) -> List[Path]:
    """
    Find all .md files in the given directory.

    Args:
        markdown_dir: Path to the directory containing markdown files.

    Returns:
        List of Path objects for markdown files.
    """
    md_path = Path(markdown_dir)
    if not md_path.exists() or not md_path.is_dir():
        logger.error("Markdown directory not found or invalid: %s", markdown_dir)
        return []

    md_files = list(md_path.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s", markdown_dir)
    return md_files


def process_markdown_files(markdown_dir: str) -> List[Document]:
    """
    Load, chunk, and prepare all markdown files in a directory for embedding.

    Args:
        markdown_dir: Path to the directory containing .md files.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    md_files = _load_markdown_files(markdown_dir)

    if not md_files:
        return []

    all_chunks: List[Document] = []
    for md_file in md_files:
        logger.info("Processing markdown file: %s", md_file.name)
        chunks = chunk_markdown_file(str(md_file))
        all_chunks.extend(chunks)
        logger.info("Created %d chunks from %s", len(chunks), md_file.name)

    logger.info("Total chunks created across all files: %d", len(all_chunks))
    return all_chunks


def ingest_documents_to_pinecone(documents: List[Document], batch_size: int = 16) -> bool:
    """
    Embed and store documents in Pinecone vector store.

    Args:
        documents: List of Document objects to embed and store.
        batch_size: Number of documents to process per batch.

    Returns:
        True if successful, False otherwise.
    """
    if not documents:
        logger.warning("No documents to ingest.")
        return False

    logger.info("Initializing Pinecone vector store...")

    try:
        vectorstore = VectorDatabases.get_vector_store()
        logger.info("Adding %d documents to the vector store...", len(documents))

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(batch)
                    break
                except (RuntimeError, ConnectionError, ValueError) as batch_error:
                    if "429" in str(batch_error) or attempt < max_retries - 1:
                        logger.warning(
                            "Rate limit on batch %d (attempt %d/%d). Backing off 10s.",
                            i,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(10)
                    else:
                        raise

            logger.info("Stored batch %d to %d.", i, i + len(batch))
            time.sleep(2)

        logger.info("Successfully embedded and stored all %d documents.", len(documents))

        # Invalidate the BM25 cache so the next retrieval rebuilds it
        invalidate_bm25_cache()

        return True

    except (RuntimeError, ConnectionError, ValueError) as exc:
        logger.error("Failed to embed and store documents: %s", exc)
        return False


def run_markdown_ingestion_pipeline(markdown_dir: str) -> bool:
    """
    Run the complete markdown ingestion pipeline.

    Args:
        markdown_dir: Path to directory containing .md files.

    Returns:
        True if pipeline completed successfully, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("Starting Markdown Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info("Markdown directory: %s", markdown_dir)
    logger.info("Pinecone index: %s", VectorDatabases.get_pinecone_index_name())

    # Step 1: Load and chunk markdown files
    logger.info("Step 1: Loading and chunking markdown files...")
    documents = process_markdown_files(markdown_dir)

    if not documents:
        logger.error("No documents to embed. Exiting.")
        return False

    # Step 2: Embed and store in Pinecone
    logger.info("Step 2: Embedding and storing in Pinecone...")
    success = ingest_documents_to_pinecone(documents)

    if success:
        logger.info("=" * 60)
        logger.info("MARKDOWN INGESTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("MARKDOWN INGESTION FAILED")
        logger.error("=" * 60)

    return success


def parse_pdf_with_docling(file_path: str, output_dir: str) -> bool:
    """Parse a PDF using Docling and persist the Markdown result.

    Args:
        file_path: Path to the source PDF.
        output_dir: Directory in which the Markdown should be written.

    Returns:
        True on success, False if Docling is unavailable or fails.
    """
    try:
        from langchain_docling import DoclingLoader  # type: ignore[import-not-found]
        logger.info("Attempting to parse %s using DoclingLoader...", file_path)
        loader = DoclingLoader(file_path=file_path)
        docs = loader.load()
        md_content = "\n\n".join(doc.page_content for doc in docs)

        return _save_markdown(file_path, output_dir, md_content)
    except ImportError:
        logger.warning("langchain_docling is not installed.")
        return False
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("Docling failed: %s", exc)
        return False


def parse_pdf_with_unstructured(file_path: str, output_dir: str) -> bool:
    """Parse a PDF using Unstructured and persist the Markdown result.

    Args:
        file_path: Path to the source PDF.
        output_dir: Directory in which the Markdown should be written.

    Returns:
        True on success, False if Unstructured is unavailable or fails.
    """
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        logger.info("Attempting to parse %s using UnstructuredPDFLoader...", file_path)
        loader = UnstructuredPDFLoader(file_path=file_path, mode="elements")
        docs = loader.load()

        md_content = ""
        for doc in docs:
            category = doc.metadata.get("category", "")
            text = doc.page_content.strip()
            if not text:
                continue
            if category == "Title":
                md_content += f"\n## {text}\n\n"
            elif category == "ListItem":
                md_content += f"- {text}\n"
            else:
                md_content += f"{text}\n\n"

        return _save_markdown(file_path, output_dir, md_content)
    except ImportError:
        logger.warning("langchain-community or unstructured is not installed.")
        return False
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("Unstructured parser failed: %s", exc)
        return False


def parse_pdf_with_pymupdf(file_path: str, output_dir: str) -> bool:
    """Parse a PDF using PyMuPDF (fitz) and persist the Markdown result.

    Args:
        file_path: Path to the source PDF.
        output_dir: Directory in which the Markdown should be written.

    Returns:
        True on success, False if PyMuPDF is unavailable or fails.
    """
    try:
        import fitz  # type: ignore[import-not-found]  # pymupdf
        logger.info("Attempting to parse %s using PyMuPDF...", file_path)
        doc = fitz.open(file_path)
        md_content = ""

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                md_content += f"\n\n## Page {page_num + 1}\n\n{text}\n"

        doc.close()
        return _save_markdown(file_path, output_dir, md_content)
    except ImportError:
        logger.warning("PyMuPDF (fitz) is not installed.")
        return False
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("PyMuPDF parser failed: %s", exc)
        return False


def process_pdf_with_fallback(pdf_path: str, output_dir: str) -> bool:
    """
    Try Docling if GPU is available, otherwise fall back to Unstructured,
    then fall back to PyMuPDF as a last resort.

    Args:
        pdf_path: Absolute path to the source PDF file.
        output_dir: Directory in which the Markdown output will be saved.

    Returns:
        True on successful conversion; False if all parsers fail.
    """
    gpu_available = False
    try:
        import torch  # type: ignore[import-not-found]
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    # Try Docling first if GPU is available
    if gpu_available:
        logger.info("GPU detected. Trying high-accuracy Docling parser.")
        if parse_pdf_with_docling(pdf_path, output_dir):
            return True

    logger.info("GPU not detected or Docling failed. Falling back to CPU parsing.")

    # Try Unstructured
    if parse_pdf_with_unstructured(pdf_path, output_dir):
        return True

    # Try PyMuPDF as final fallback
    logger.info("Trying PyMuPDF as final fallback parser.")
    if parse_pdf_with_pymupdf(pdf_path, output_dir):
        return True

    logger.error("All PDF parsers failed for %s", pdf_path)
    return False


def _save_markdown(file_path: str, output_dir: str, md_content: str) -> bool:
    """Write parsed markdown content to a sibling .md file.

    Args:
        file_path: Original source PDF path (used to derive the .md filename).
        output_dir: Directory in which the .md will be written.
        md_content: Markdown content to persist.

    Returns:
        True on successful write, False on I/O error.
    """
    path_obj = Path(file_path)
    output_path = Path(output_dir)
    md_file_path = output_path / f"{path_obj.stem}.md"
    try:
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Successfully saved Markdown to %s", md_file_path)
        return True
    except IOError as exc:
        logger.error("Failed to write Markdown file %s: %s", md_file_path, exc)
        return False
