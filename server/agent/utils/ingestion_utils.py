"""Helpers for ingesting legal PDF documents."""

from pathlib import Path

from agent.utils.logger import get_logger

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


def parse_pdf_with_docling(file_path: str, output_dir: str) -> bool:
    """Parse a PDF using Docling and persist the Markdown result.

    Args:
        file_path: Path to the source PDF.
        output_dir: Directory in which the Markdown should be written.

    Returns:
        True on success, False if Docling is unavailable or fails.
    """
    try:
        from langchain_docling import DoclingLoader
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
