"""PDF parsing utilities with multiple backend fallback support."""

import time
from pathlib import Path
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PDFParserService:
    """Service for parsing PDF files using multiple backend fallbacks."""

    def __init__(
        self,
        prefer_docling: bool = True,
    ) -> None:
        """Initialize the PDF parser service.

        Args:
            prefer_docling: Whether to try Docling first (requires GPU).
        """
        self._prefer_docling = prefer_docling

    def _has_gpu(self) -> bool:
        """Check if GPU is available for Docling."""
        try:
            import torch  # type: ignore[import-not-found]
            return torch.cuda.is_available()
        except ImportError:
            return False

    def parse_pdf_with_docling(self, file_path: str, output_dir: str) -> bool:
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

            return self._save_markdown(file_path, output_dir, md_content)
        except ImportError:
            logger.warning("langchain_docling is not installed.")
            return False
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Docling failed: %s", exc)
            return False

    def parse_pdf_with_unstructured(self, file_path: str, output_dir: str) -> bool:
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

            return self._save_markdown(file_path, output_dir, md_content)
        except ImportError:
            logger.warning("langchain-community or unstructured is not installed.")
            return False
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Unstructured parser failed: %s", exc)
            return False

    def parse_pdf_with_pymupdf(self, file_path: str, output_dir: str) -> bool:
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
            return self._save_markdown(file_path, output_dir, md_content)
        except ImportError:
            logger.warning("PyMuPDF (fitz) is not installed.")
            return False
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("PyMuPDF parser failed: %s", exc)
            return False

    def _save_markdown(self, file_path: str, output_dir: str, md_content: str) -> bool:
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

    def process_pdf_with_fallback(self, pdf_path: str, output_dir: str) -> bool:
        """
        Try Docling if GPU is available, otherwise fall back to Unstructured,
        then fall back to PyMuPDF as a last resort.

        Args:
            pdf_path: Absolute path to the source PDF file.
            output_dir: Directory in which the Markdown output will be saved.

        Returns:
            True on successful conversion; False if all parsers fail.
        """
        gpu_available = self._has_gpu()

        # Try Docling first if GPU is available and preferred
        if gpu_available and self._prefer_docling:
            logger.info("GPU detected. Trying high-accuracy Docling parser.")
            if self.parse_pdf_with_docling(pdf_path, output_dir):
                return True

        logger.info("GPU not detected or Docling failed. Falling back to CPU parsing.")

        # Try Unstructured
        if self.parse_pdf_with_unstructured(pdf_path, output_dir):
            return True

        # Try PyMuPDF as final fallback
        logger.info("Trying PyMuPDF as final fallback parser.")
        if self.parse_pdf_with_pymupdf(pdf_path, output_dir):
            return True

        logger.error("All PDF parsers failed for %s", pdf_path)
        return False


# Backward compatible functions
def parse_pdf_with_docling(file_path: str, output_dir: str) -> bool:
    """Backward compatible function."""
    service = PDFParserService()
    return service.parse_pdf_with_docling(file_path, output_dir)


def parse_pdf_with_unstructured(file_path: str, output_dir: str) -> bool:
    """Backward compatible function."""
    service = PDFParserService()
    return service.parse_pdf_with_unstructured(file_path, output_dir)


def parse_pdf_with_pymupdf(file_path: str, output_dir: str) -> bool:
    """Backward compatible function."""
    service = PDFParserService()
    return service.parse_pdf_with_pymupdf(file_path, output_dir)


def process_pdf_with_fallback(pdf_path: str, output_dir: str) -> bool:
    """Backward compatible function."""
    service = PDFParserService()
    return service.process_pdf_with_fallback(pdf_path, output_dir)