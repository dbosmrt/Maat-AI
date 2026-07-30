"""Markdown text chunking utilities with OOP design."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MarkdownChunker:
    """Chunks markdown files based on headers with size enforcement."""

    def __init__(
        self,
        chunk_size: int = 2500,
        chunk_overlap: int = 400,
        headers_to_split_on: List[tuple[str, str]] | None = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.
            headers_to_split_on: List of (markdown_symbol, header_name) tuples.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        self._headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers_to_split_on,
            strip_headers=False,
        )

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def chunk_markdown_file(self, file_path: str) -> List[Document]:
        """
        Read a Markdown file and chunk it based on headers.

        Falls back to recursive character splitting if sections are too large.

        Args:
            file_path: Path to the markdown file.

        Returns:
            List of chunked Document objects with metadata.
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.error("Markdown file not found: %s", file_path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except IOError as exc:
            logger.error("Failed to read file %s: %s", file_path, exc)
            return []

        # 1. Primary Strategy: Semantic Header Splitter
        header_splits = self._markdown_splitter.split_text(content)

        # 2. Secondary/Fallback Strategy: Size-Enforcement Splitter
        final_chunks = self._text_splitter.split_documents(header_splits)

        # Add metadata about the source file and hierarchy
        for chunk in final_chunks:
            doc_name = path.stem
            chunk.metadata["source"] = path.name
            chunk.metadata["document"] = doc_name

            # Build hierarchical context string
            hierarchy = [doc_name]
            for i in range(1, 7):
                header_key = f"Header {i}"
                if header_key in chunk.metadata:
                    hierarchy.append(chunk.metadata[header_key].strip())

            chunk.metadata["context_path"] = " - ".join(hierarchy)

        logger.info("Chunked %s into %d pieces.", path.name, len(final_chunks))
        return final_chunks


# Backward compatible function
_chunker: MarkdownChunker | None = None


def get_chunker() -> MarkdownChunker:
    """Get or create the global chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = MarkdownChunker()
    return _chunker


def chunk_markdown_file(file_path: str) -> List[Document]:
    """Backward compatible function."""
    return get_chunker().chunk_markdown_file(file_path)