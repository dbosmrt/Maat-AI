"""
Integration tests for the ingestion node using real PDFs.
"""

import sys
import os

# Ensure server module can be imported before other imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../server')))

from pathlib import Path
import pytest

from agent.utils.pdf_parser import process_pdf_with_fallback

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data"
MD_DATA_DIR = BASE_DIR / "data" / "markdown"

# Skip integration tests if unstructured is not available (CI environment)
try:
    import unstructured
    import langchain_community
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

@pytest.mark.skipif(not UNSTRUCTURED_AVAILABLE, reason="unstructured and langchain_community not installed")
@pytest.mark.parametrize("pdf_filename", [
    "BNS.pdf",
    "BSA.pdf",
    "BNSS.pdf"
])
def test_real_pdf_conversion(pdf_filename):
    """
    Integration test to verify real PDF conversion to Markdown.
    It takes the specified PDFs one by one and converts them.
    """
    pdf_path = RAW_DATA_DIR / pdf_filename

    # Skip if file doesn't exist
    if not pdf_path.exists():
        pytest.skip(f"Test file {pdf_path} not found.")

    # Make sure output directory exists
    MD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nAttempting to convert: {pdf_path.name}")

    # Run the conversion
    result = process_pdf_with_fallback(str(pdf_path), str(MD_DATA_DIR))

    # Assert successful return from function
    assert result is True

    # Assert the markdown file was actually created in data/markdown
    md_file_path = MD_DATA_DIR / f"{pdf_path.stem}.md"
    assert md_file_path.exists(), f"Markdown file was not created at {md_file_path}"

    # Assert it actually wrote data to the file
    file_size = md_file_path.stat().st_size
    print(f"Success! Created {md_file_path.name} (Size: {file_size} bytes)")
    assert file_size > 0, "The generated markdown file is empty!"
