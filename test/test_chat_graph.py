import pytest
import logging
from agent.utils.logger import get_logger
from agent.chat_graph import build_chat_graph


logger = get_logger(__name__)

def test_chat_graph_compilation():
    """
    Tests if the LangGraph workflow compiles successfully without routing errors.
    """
    logger.info("--- Testing Chat Graph Compilation ---")
    
    try:
        app = build_chat_graph()
        assert app is not None, "Compiled graph should not be None."
        logger.info("Graph compiled successfully!")
    except Exception as e:
        pytest.fail(f"Graph compilation failed with exception: {e}")
