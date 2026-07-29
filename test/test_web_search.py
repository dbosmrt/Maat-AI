import pytest
import logging
from agent.utils.logger import get_logger
from agent.state import AgentState
from agent.node.web_search import web_search_node


logger = get_logger(__name__)

def test_web_search_node():
    """Test that the web search node can fetch live results from DuckDuckGo."""
    state = AgentState(
        query="Arnesh Kumar vs State of Bihar",
        requires_case_law=True,
        session_id="test", chat_history=[], memory_summary="", documents=[], case_laws=[], generation="", iteration_count=0
    )

    logger.info("--- Testing Web Search Node ---")
    result = web_search_node(state)

    assert "case_laws" in result
    case_laws = result["case_laws"]

    assert isinstance(case_laws, list)
    assert len(case_laws) > 0, "Web search returned zero results!"

    # Verify format
    first_result = case_laws[0]
    assert "[External Source:" in first_result, "Web search result missing format prefix."

    logger.info(f"Top result preview:\n{first_result[:200]}...")
