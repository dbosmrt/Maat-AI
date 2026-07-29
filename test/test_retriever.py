import pytest
import logging
from unittest.mock import patch, MagicMock
from agent.utils.logger import get_logger
from agent.state import AgentState
from agent.node.retriever import retriever_node

logger = get_logger(__name__)

@patch('agent.node.retriever.VectorDatabases.get_vector_store')
@patch('agent.node.retriever._get_bm25_retriever')
def test_retriever_node_basic(mock_get_bm25, mock_get_vs):
    """
    Test to ensure the retriever node can retrieve and fuse results correctly.
    """
    query = "What is the definition of a bailable offence?"

    # Mock Pinecone dense vectorstore
    mock_vs = MagicMock()
    mock_dense_retriever = MagicMock()
    mock_dense_doc = MagicMock()
    mock_dense_doc.page_content = "Dense document content"
    mock_dense_doc.metadata = {"context_path": "BNS.pdf"}
    mock_dense_retriever.invoke.return_value = [mock_dense_doc]
    mock_vs.as_retriever.return_value = mock_dense_retriever
    mock_get_vs.return_value = mock_vs

    # Mock BM25 sparse retriever
    mock_bm25 = MagicMock()
    mock_sparse_doc = MagicMock()
    mock_sparse_doc.page_content = "Sparse document content"
    mock_sparse_doc.metadata = {"context_path": "BNSS.pdf"}
    mock_bm25.invoke.return_value = [mock_sparse_doc]
    mock_get_bm25.return_value = mock_bm25

    # Mock state
    state = AgentState(
        query=query,
        session_id="test_session",
        chat_history=[],
        memory_summary="",
        documents=[],
        case_laws=[],
        generation="",
        iteration_count=0,
        decomposed_query={}
    )

    # Execute node
    result = retriever_node(state)

    # Validate output
    assert "documents" in result, "Retriever did not return the 'documents' key."
    docs = result["documents"]

    assert isinstance(docs, list), "Documents should be a list"
    assert len(docs) == 2, "Retriever should return fused chunks."

    # Verify formatting
    assert "[Source:" in docs[0], "Retrieved chunk string does not contain metadata."
