"""Markdown text cleaning utilities with OOP design."""

import re

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MarkdownCleaner:
    """Cleans and normalizes markdown text from Indian Legal PDFs."""

    def __init__(self) -> None:
        """Initialize the cleaner with default patterns."""
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> dict[str, tuple[str, str, int]]:
        """Build the regex patterns for cleaning.

        Returns:
            Dictionary of pattern_name -> (pattern, replacement, flags)
        """
        return {
            "non_ascii": (
                r"[^\x00-\x7F]+",
                " ",
                0,
            ),
            "chapter_to_h1": (
                r"^##\s*(CHAPTER\s+[A-Z0-9]+)",
                r"# \1",
                re.MULTILINE | re.IGNORECASE,
            ),
            "illustration_to_bold": (
                r"^##\s*(Illustration[s]?\.?|Explanation[s]?\.?)",
                r"**\1**",
                re.MULTILINE | re.IGNORECASE,
            ),
            "section_to_h2": (
                r"^(\d{1,4})\.\s",
                r"## Section \1.\n\n",
                re.MULTILINE,
            ),
            "page_slashes": (
                r"/{4,}",
                "",
                0,
            ),
            "page_underscores": (
                r"_{4,}",
                "",
                0,
            ),
            "excessive_newlines": (
                r"\n{3,}",
                "\n\n",
                0,
            ),
        }

    def clean(self, text: str) -> str:
        """
        Clean the raw markdown text generated from Indian Legal PDFs.

        Args:
            text: The raw markdown content.

        Returns:
            The cleaned and structurally corrected markdown.
        """
        if not text:
            return ""

        # Apply each pattern in order
        for pattern_name, (pattern, replacement, flags) in self._patterns.items():
            text = re.sub(pattern, replacement, text, flags=flags)

        return text

    def clean_file(self, file_path: str, encoding: str = "utf-8") -> bool:
        """
        Clean a markdown file in-place.

        Args:
            file_path: Path to the .md file.
            encoding: File encoding.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            cleaned_content = self.clean(content)

            with open(file_path, "w", encoding=encoding) as f:
                f.write(cleaned_content)

            logger.info("Cleaned and formatted: %s", file_path)
            return True
        except (IOError, UnicodeDecodeError) as exc:
            logger.error("Failed to clean file %s: %s", file_path, exc)
            return False


# Backward compatible function
_cleaner: MarkdownCleaner | None = None


def get_cleaner() -> MarkdownCleaner:
    """Get or create the global cleaner instance."""
    global _cleaner
    if _cleaner is None:
        _cleaner = MarkdownCleaner()
    return _cleaner


def clean_markdown_text(text: str) -> str:
    """Backward compatible function."""
    return get_cleaner().clean(text)