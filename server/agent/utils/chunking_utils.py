"""Markdown text chunking utilities for the Legal RAG Chatbot."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from agent.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_markdown_file(file_path: str) -> List[Document]:
    """
    Read a Markdown file and chunk it based on headers.

    Falls back to recursive character splitting if sections are too large.
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

    # Define headers to split on. Legal docs often use ## for sections.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    # 1. Primary Strategy: Semantic Header Splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    header_splits = markdown_splitter.split_text(content)

    # 2. Secondary/Fallback Strategy: Size-Enforcement Splitter
    # If the markdown splitter fails to find headers OR a specific section
    # between two headers is still too large, this cuts it down.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=400,
    )

    final_chunks = text_splitter.split_documents(header_splits)

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
