"""
Pytest suite for the ingestion node, covering fallback logic and LangGraph state processing.
"""

import sys
import os

# Ensure server module can be imported before other imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../server')))

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from agent.node.ingestion import ingestion_node
from agent.utils.pdf_parser import process_pdf_with_fallback

@pytest.fixture
def mock_state():
    return {
        "ingest_input_dir": "/fake/input",
        "ingest_output_dir": "/fake/output"
    }


@pytest.fixture
def mock_torch_gpu(monkeypatch):
    """Mock torch with GPU available for ingestion tests."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, 'torch', mock_torch)


@pytest.fixture
def mock_torch_cpu(monkeypatch):
    """Mock torch as unavailable for ingestion tests testing CPU fallback."""
    # Replace torch in sys.modules with a mock that raises ImportError on attribute access
    class MockTorch:
        def __getattr__(self, name):
            raise ImportError("torch not available")
    monkeypatch.setitem(sys.modules, 'torch', MockTorch())


@patch('agent.utils.ingestion_utils.parse_pdf_with_docling')
@patch('agent.utils.ingestion_utils.parse_pdf_with_unstructured')
def test_process_pdf_with_fallback_docling_success(mock_unstructured, mock_docling, mock_torch_gpu):
    """Test that docling success skips fallbacks when GPU is available."""
    mock_docling.return_value = True
    mock_unstructured.return_value = True

    result = process_pdf_with_fallback("/fake/test.pdf", "/fake/out")

    assert result is True
    mock_docling.assert_called_once_with("/fake/test.pdf", "/fake/out")
    mock_unstructured.assert_not_called()


@patch('agent.utils.ingestion_utils.parse_pdf_with_docling')
@patch('agent.utils.ingestion_utils.parse_pdf_with_unstructured')
def test_process_pdf_with_fallback_unstructured_success(mock_unstructured, mock_docling, mock_torch_cpu):
    """Test that unstructured is used if GPU is not available."""
    mock_unstructured.return_value = True

    result = process_pdf_with_fallback("/fake/test.pdf", "/fake/out")

    assert result is True
    mock_docling.assert_not_called()
    mock_unstructured.assert_called_once_with("/fake/test.pdf", "/fake/out")


@patch('agent.utils.ingestion_utils.parse_pdf_with_docling')
@patch('agent.utils.ingestion_utils.parse_pdf_with_unstructured')
def test_process_pdf_with_fallback_docling_fails(mock_unstructured, mock_docling, mock_torch_gpu):
    """Test fallback to unstructured when Docling fails (returns False)."""
    mock_docling.return_value = False
    mock_unstructured.return_value = True

    result = process_pdf_with_fallback("/fake/test.pdf", "/fake/out")

    assert result is True
    mock_docling.assert_called_once_with("/fake/test.pdf", "/fake/out")
    mock_unstructured.assert_called_once_with("/fake/test.pdf", "/fake/out")


@patch('agent.node.ingestion.validate_paths')
@patch('agent.node.ingestion.Path.glob')
@patch('agent.node.ingestion.process_pdf_with_fallback')
def test_ingestion_node_success(mock_process, mock_glob, mock_validate, mock_state):
    """Test that the LangGraph node successfully processes the state."""
    mock_validate.return_value = True
    mock_glob.return_value = [Path("/fake/input/doc1.pdf"), Path("/fake/input/doc2.pdf")]
    mock_process.return_value = True

    result = ingestion_node(mock_state)

    mock_validate.assert_called_once_with("/fake/input", "/fake/output")
    assert mock_process.call_count == 2
    mock_process.assert_any_call("/fake/input/doc1.pdf", "/fake/output")
    mock_process.assert_any_call("/fake/input/doc2.pdf", "/fake/output")
    assert result == {"ingest_status": "Completed: Successfully converted 2/2 files."}


def test_ingestion_node_missing_dirs():
    """Test that the LangGraph node fails cleanly if state lacks required directories."""
    state = {}
    result = ingestion_node(state)
    assert result == {"ingest_status": "Failed: Missing directories in state"}
