import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from agent.state import AgentState
from agent.node.chunking import chunking_node
from agent.node.embedding import embedding_node

def test_chunk_and_embed_pipeline():
    """
    Test chunking and embedding nodes correctly pass data and call Pinecone without real API requests.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock markdown file
        md_file = Path(temp_dir) / "BSA.md"
        md_file.write_text("# Title\\n\\nSome test content.\\n", encoding="utf-8")

        state = AgentState(
            pdf_dir="data",
            ingest_output_dir=temp_dir,
            chunk_output_dir="data/chunks",
            db_dir="vector_store",
            messages=[]
        )

        # 1. Execute Chunking Node
        chunk_result = chunking_node(state)

        assert "documents" in chunk_result, "Chunking node did not return documents"
        chunks = chunk_result["documents"]
        assert len(chunks) > 0, "No chunks were generated"

        # Update State
        state["documents"] = chunks

        # 2. Mock Vector Database and execute Embedding Node
        with patch('agent.node.embedding.VectorDatabases.get_vector_store') as mock_get_vs:
            mock_vs = MagicMock()
            mock_get_vs.return_value = mock_vs

            embed_result = embedding_node(state)

            # Assert successful execution and mock calls
            assert "Completed Successfully" in embed_result.get("ingest_status", "")
            mock_vs.add_documents.assert_called_once()
