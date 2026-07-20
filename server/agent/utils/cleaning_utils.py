"""Helpers for cleaning markdown text for the Legal RAG Chatbot."""

import re


def clean_markdown_text(text: str) -> str:
    """
    Clean the raw markdown text generated from Indian Legal PDFs.

    Args:
        text: The raw markdown content.

    Returns:
        The cleaned and structurally corrected markdown.
    """
    # 1. Remove non-ASCII garbage (e.g., garbled Hindi fonts)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 2. Upgrade CHAPTER to main heading (#) to establish hierarchy over Sections (##)
    text = re.sub(
        r"^##\s*(CHAPTER\s+[A-Z0-9]+)",
        r"# \1",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 3. Demote Illustrations and Explanations so they don't break chunking
    # e.g. "## Illustration." -> "**Illustration.**"
    text = re.sub(
        r"^##\s*(Illustration[s]?\.?|Explanation[s]?\.?)",
        r"**\1**",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 4. Promote section numbers to proper markdown headers (##)
    # e.g., "1. (1) This Act..." -> "## Section 1.\n\n(1) This Act..."
    # Matches lines starting with 1 to 4 digits followed by a period and a space
    text = re.sub(r"^(\d{1,4})\.\s", r"## Section \1.\n\n", text, flags=re.MULTILINE)

    # 5. Remove page ending slashes (e.g. ////) or underscores (____) left by unstructured parsers
    text = re.sub(r"/{4,}", "", text)
    text = re.sub(r"_{4,}", "", text)
    text = re.sub(r"(?:\_){4,}", "", text)

    # 6. Remove excessive newlines that might cause empty chunks
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text
