"""Helpers for ingesting legal PDF documents and markdown files."""

import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from agent.utils.chunking_utils import chunk_markdown_file
from agent.utils.embedding_utils import VectorDatabaseService
from agent.utils.logger import get_logger
from agent.node.retriever import invalidate_bm25_cache

logger = get_logger(__name__)


class DocumentIngestionService:
    """Service for document ingestion pipeline with dependency injection."""

    def __init__(
        self,
        vector_databases: VectorDatabaseService | None = None,
        batch_size: int = 16,
        max_retries: int = 3,
        retry_delay: int = 10,
        rate_limit_sleep: int = 2,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            vector_databases: Vector database facade (injected for testing).
            batch_size: Number of documents per batch for embedding.
            max_retries: Maximum retries for rate-limited operations.
            retry_delay: Delay in seconds between retries.
            rate_limit_sleep: Delay between batches to respect rate limits.
        """
        self._vector_databases = vector_databases or VectorDatabaseService()
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._rate_limit_sleep = rate_limit_sleep

    def validate_paths(self, input_dir: str, output_dir: str) -> bool:
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

    def _load_markdown_files(self, markdown_dir: str) -> List[Path]:
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

    def process_markdown_files(self, markdown_dir: str) -> List[Document]:
        """
        Load, chunk, and prepare all markdown files in a directory for embedding.

        Args:
            markdown_dir: Path to the directory containing .md files.

        Returns:
            List of chunked Document objects ready for embedding.
        """
        md_files = self._load_markdown_files(markdown_dir)

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

    def ingest_documents_to_pinecone(self, documents: List[Document]) -> bool:
        """
        Embed and store documents in Pinecone vector store.

        Args:
            documents: List of Document objects to embed and store.

        Returns:
            True if successful, False otherwise.
        """
        if not documents:
            logger.warning("No documents to ingest.")
            return False

        logger.info("Initializing Pinecone vector store...")

        try:
            vectorstore = self._vector_databases.get_vector_store()
            logger.info("Adding %d documents to the vector store...", len(documents))

            for i in range(0, len(documents), self._batch_size):
                batch = documents[i : i + self._batch_size]

                for attempt in range(self._max_retries):
                    try:
                        vectorstore.add_documents(batch)
                        break
                    except (RuntimeError, ConnectionError, ValueError) as batch_error:
                        if "429" in str(batch_error) or attempt < self._max_retries - 1:
                            logger.warning(
                                "Rate limit on batch %d (attempt %d/%d). Backing off %ds.",
                                i,
                                attempt + 1,
                                self._max_retries,
                                self._retry_delay,
                            )
                            time.sleep(self._retry_delay)
                        else:
                            raise

                logger.info("Stored batch %d to %d.", i, i + len(batch))
                time.sleep(self._rate_limit_sleep)

            logger.info("Successfully embedded and stored all %d documents.", len(documents))

            # Invalidate the BM25 cache so the next retrieval rebuilds it
            invalidate_bm25_cache()

            return True

        except (RuntimeError, ConnectionError, ValueError) as exc:
            logger.error("Failed to embed and store documents: %s", exc)
            return False

    def run_markdown_ingestion_pipeline(self, markdown_dir: str) -> bool:
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
        logger.info("Pinecone index: %s", self._vector_databases.get_pinecone_index_name())

        # Step 1: Load and chunk markdown files
        logger.info("Step 1: Loading and chunking markdown files...")
        documents = self.process_markdown_files(markdown_dir)

        if not documents:
            logger.error("No documents to embed. Exiting.")
            return False

        # Step 2: Embed and store in Pinecone
        logger.info("Step 2: Embedding and storing in Pinecone...")
        success = self.ingest_documents_to_pinecone(documents)

        if success:
            logger.info("=" * 60)
            logger.info("MARKDOWN INGESTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("MARKDOWN INGESTION FAILED")
            logger.error("=" * 60)

        return success


# Backward compatible functions
def validate_paths(input_dir: str, output_dir: str) -> bool:
    """Backward compatible function."""
    service = DocumentIngestionService()
    return service.validate_paths(input_dir, output_dir)


def process_markdown_files(markdown_dir: str) -> List[Document]:
    """Backward compatible function."""
    service = DocumentIngestionService()
    return service.process_markdown_files(markdown_dir)


def ingest_documents_to_pinecone(documents: List[Document], batch_size: int = 16) -> bool:
    """Backward compatible function."""
    service = DocumentIngestionService(batch_size=batch_size)
    return service.ingest_documents_to_pinecone(documents)


def run_markdown_ingestion_pipeline(markdown_dir: str) -> bool:
    """Backward compatible function."""
    service = DocumentIngestionService()
    return service.run_markdown_ingestion_pipeline(markdown_dir)
