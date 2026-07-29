import os
import pytest
from pathlib import Path

from agent.node.chunking import chunk_markdown_file

def test_chunking_all_files():
    """
    Test that reads all markdown files from data/markdown,
    chunks them using the header-based strategy, prints the number of chunks,
    and saves the chunks to a text file in data/chunks for manual inspection.
    """
    md_dir = Path("data/markdown")
    chunk_dir = Path("data/chunks")

    # Check if files exist
    if not md_dir.exists() or not list(md_dir.glob("*.md")):
        pytest.skip(f"No markdown files found in {md_dir} to chunk.")

    # Ensure output directory exists
    chunk_dir.mkdir(parents=True, exist_ok=True)

    md_files = list(md_dir.glob("*.md"))

    print("\n--- CHUNKING TEST RESULTS ---")

    for md_file in md_files:
        print(f"\nProcessing {md_file.name}...")

        # Call the chunking function directly
        chunks = chunk_markdown_file(str(md_file))

        # Print number of chunks
        print(f"Total chunks created for {md_file.name}: {len(chunks)}")

        assert len(chunks) > 0, f"No chunks were generated for {md_file.name}"

        # Save chunks to a file for manual inspection
        output_file = chunk_dir / f"{md_file.stem}_chunks_debug.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i+1} ===\n")
                f.write(f"METADATA: {chunk.metadata}\n")
                f.write(f"CONTENT:\n{chunk.page_content}\n")
                f.write("=" * 80 + "\n\n")

        print(f"Saved chunk debug output to {output_file}")
