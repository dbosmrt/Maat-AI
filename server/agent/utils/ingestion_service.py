"""Document ingestion service with proper OOP design."""

import time
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from ..utils.embedding_utils import VectorDatabaseService
from ..utils.pdf_parser import PDFParserService
from ..utils.chunking_utils import MarkdownChunker
from ..utils.cleaning_utils import MarkdownCleaner
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIngestionService:
    """Service for complete document ingestion pipeline (PDF → Markdown → Chunk → Embed)."""

    def __init__(
        self,
        vector_database_service: Optional[VectorDatabaseService] = None,
        pdf_parser: Optional[PDFParserService] = None,
        chunker: Optional[MarkdownChunker] = None,
        cleaner: Optional[MarkdownCleaner] = None,
        batch_size: int = 16,
        max_retries: int = 3,
        retry_delay_seconds: int = 10,
        rate_limit_sleep_seconds: int = 2,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            vector_database_service: Vector database service. If None, creates default.
            pdf_parser: PDF parser service. If None, creates default.
            chunker: Markdown chunker. If None, creates default.
            cleaner: Markdown cleaner. If None, creates default.
            batch_size: Number of documents per batch for upsert.
            max_retries: Maximum retries for rate-limited operations.
            retry_delay_seconds: Delay between retries in seconds.
            rate_limit_sleep_seconds: Sleep between batches in seconds.
        """
        self._vector_database_service = vector_database_service or VectorDatabaseService()
        self._pdf_parser = pdf_parser or PDFParserService()
        self._chunker = chunker or MarkdownChunker()
        self._cleaner = cleaner or MarkdownCleaner()
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._rate_limit_sleep = rate_limit_sleep_seconds

    def validate_paths(self, input_dir: str, output_dir: str) -> bool:
        """Check that the input directory exists and create the output dir.

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

    def convert_pdf_to_markdown(self, input_dir: str, output_dir: str) -> int:
        """Convert all PDFs in input directory to Markdown in output directory.

        Args:
            input_dir: Directory containing source PDFs.
            output_dir: Directory for Markdown output.

        Returns:
            Number of successfully converted files.
        """
        if not self.validate_paths(input_dir, output_dir):
            return 0

        pdf_files = list(Path(input_dir).glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", input_dir)
            return 0

        logger.info("Found %d PDF(s). Starting conversion...", len(pdf_files))
        success_count = 0

        for pdf_file in pdf_files:
            if self._pdf_parser.process_pdf_with_fallback(str(pdf_file), output_dir):
                success_count += 1

        logger.info("Successfully converted %d/%d files.", success_count, len(pdf_files))
        return success_count

    def clean_markdown_directory(self, markdown_dir: str) -> int:
        """Clean all Markdown files in a directory.

        Args:
            markdown_dir: Directory containing .md files to clean.

        Returns:
            Number of files cleaned.
        """
        md_path = Path(markdown_dir)
        if not md_path.exists() or not md_path.is_dir():
            logger.error("Markdown directory not found: %s", markdown_dir)
            return 0

        md_files = list(md_path.glob("*.md"))
        if not md_files:
            logger.warning("No .md files found in %s", markdown_dir)
            return 0

        cleaned_count = 0
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                cleaned_content = self._cleaner.clean_markdown_text(content)

                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)

                logger.info("Cleaned and formatted: %s", md_file.name)
                cleaned_count += 1
            except (IOError, UnicodeDecodeError) as exc:
                logger.error("Failed to clean file %s: %s", md_file.name, exc)

        return cleaned_count

    def chunk_markdown_directory(self, markdown_dir: str) -> List[Document]:
        """Chunk all Markdown files in a directory.

        Args:
            markdown_dir: Directory containing .md files to chunk.

        Returns:
            List of chunked Document objects.
        """
        md_path = Path(markdown_dir)
        if not md_path.exists() or not md_path.is_dir():
            logger.error("Markdown directory not found: %s", markdown_dir)
            return []

        md_files = list(md_path.glob("*.md"))
        if not md_files:
            logger.warning("No .md files found in %s", markdown_dir)
            return []

        all_chunks: List[Document] = []
        for md_file in md_files:
            chunks = self._chunker.chunk_markdown_file(str(md_file))
            all_chunks.extend(chunks)
            logger.info("Created %d chunks from %s", len(chunks), md_file.name)

        logger.info("Total chunks created across all files: %d", len(all_chunks))
        return all_chunks

    def embed_and_store(self, documents: List[Document]) -> bool:
        """Embed and store documents in Pinecone.

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
            vectorstore = self._vector_database_service.get_vector_store()
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

            # Invalidate the BM25 cache so next retrieval rebuilds it
            from ..node.retriever import invalidate_bm25_cache
            invalidate_bm25_cache()

            return True

        except (RuntimeError, ConnectionError, ValueError) as exc:
            logger.error("Failed to embed and store documents: %s", exc)
            return False

    def run_full_pipeline(self, markdown_dir: str) -> bool:
        """Run the complete Markdown ingestion pipeline.

        Args:
            markdown_dir: Path to directory containing .md files.

        Returns:
            True if pipeline completed successfully, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("Starting Markdown Ingestion Pipeline")
        logger.info("=" * 60)
        logger.info("Markdown directory: %s", markdown_dir)
        logger.info("Pinecone index: %s", self.get_pinecone_index_name())

        # Step 1: Load and chunk markdown files
        logger.info("Step 1: Loading and chunking markdown files...")
        documents = self.chunk_markdown_directory(markdown_dir)

        if not documents:
            logger.error("No documents to embed. Exiting.")
            return False

        # Step 2: Embed and store in Pinecone
        logger.info("Step 2: Embedding and storing in Pinecone...")
        success = self.embed_and_store(documents)

        if success:
            logger.info("=" * 60)
            logger.info("MARKDOWN INGESTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("MARKDOWN INGESTION FAILED")
            logger.error("=" * 60)

        return success

    def get_pinecone_index_name(self) -> str:
        """Get the Pinecone index name from the vector database service."""
        return self._vector_database_service.get_pinecone_index_name()


# Backward compatible functions
_ingestion_service: Optional[DocumentIngestionService] = None


def get_ingestion_service() -> DocumentIngestionService:
    """Get or create the global ingestion service instance."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DocumentIngestionService()
    return _ingestion_service


def validate_paths(input_dir: str, output_dir: str) -> bool:
    """Backward compatible function."""
    return get_ingestion_service().validate_paths(input_dir, output_dir)


def process_markdown_files(markdown_dir: str) -> List[Document]:
    """Backward compatible function."""
    return get_ingestion_service().chunk_markdown_directory(markdown_dir)


def ingest_documents_to_pinecone(documents: List[Document], batch_size: int = 16) -> bool:
    """Backward compatible function."""
    service = get_ingestion_service()
    original_batch = service._batch_size
    service._batch_size = batch_size
    try:
        return service.embed_and_store(documents)
    finally:
        service._batch_size = original_batch


def run_markdown_ingestion_pipeline(markdown_dir: str) -> bool:
    """Backward compatible function."""
    return get_ingestion_service().run_full_pipeline(markdown_dir)
