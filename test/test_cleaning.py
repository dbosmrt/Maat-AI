import pytest
from pathlib import Path

from agent.state import AgentState
from agent.node.cleaning import clean_markdown_text, cleaning_node

def test_clean_markdown_text_unit():
    """
    Unit test to verify the specific regex rules of the markdown cleaner.
    """
    raw_text = """
## CHAPTER I
PRELIMINARY
## vlk/kkj.k

1. ( 1 ) This Act may be called the Bharatiya Nyaya Sanhita, 2023.

## Illustration.
A commits a crime.

## Explanation.
This is an explanation.

////
End of page.
"""
    cleaned = clean_markdown_text(raw_text)

    # Check that CHAPTER was promoted
    assert "# CHAPTER I" in cleaned
    assert "## CHAPTER I" not in cleaned

    # Check that section was promoted and text dropped to next line
    assert "## Section 1.\n\n( 1 ) This Act may be called" in cleaned

    # Check that Illustration and Explanation were demoted
    assert "## Illustration." not in cleaned
    assert "**Illustration.**" in cleaned
    assert "## Explanation." not in cleaned
    assert "**Explanation.**" in cleaned

    # Check that slashes were removed
    assert "////" not in cleaned


def test_cleaning_node_integration():
    """
    Integration test that runs the cleaning node on the actual markdown files
    in data/markdown, cleans them, and overrides them.
    """
    md_dir = "data/markdown"

    md_path = Path(md_dir)
    if not md_path.exists() or not list(md_path.glob("*.md")):
        pytest.skip(f"No markdown files found in {md_dir} to run integration test.")

    # Setup mock AgentState pointing to the actual markdown directory
    state = AgentState(
        pdf_dir="data",
        ingest_output_dir=md_dir,
        chunk_output_dir="data/chunks",
        db_dir="data/chroma_db",
        messages=[]
    )

    # Run the cleaning node
    result = cleaning_node(state)

    # Assert successful execution
    assert "Completed" in result.get("ingest_status", "")

    # Read the actual modified files and ensure our regex successfully cleaned them
    md_files = list(md_path.glob("*.md"))
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # The raw Unstructured PDF loader outputs "## Illustration.",
        # which must no longer exist anywhere in the cleaned files!
        assert "## Illustration." not in content, f"Failed to demote Illustration in {md_file.name}"
        assert "## Explanation." not in content, f"Failed to demote Explanation in {md_file.name}"
