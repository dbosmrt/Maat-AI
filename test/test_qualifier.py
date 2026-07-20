import pytest
import logging
from agent.utils.logger import get_logger
from agent.state import AgentState
from agent.node.qualifier import qualifier_node


logger = get_logger(__name__)

@pytest.mark.parametrize("query, expected_domain, expected_scenario, expected_case_law", [
    (
        "What is the punishment for theft under the new Bharatiya Nyaya Sanhita?",
        "Criminal", False, False
    ),
    (
        "Can you provide me with Supreme Court judgments regarding anticipatory bail in dowry cases?",
        "Criminal", False, True
    ),
    (
        "If a person signs a contract under coercion, is it valid? Give me some case studies.",
        "Civil", True, True
    ),
    (
        "My neighbor built a fence on my property and is refusing to take it down. What can I do?",
        "Civil", True, False
    ),
    (
        "Who is the current Chief Justice of India?",
        "General", False, False
    )
])
def test_qualifier_node(query, expected_domain, expected_scenario, expected_case_law):
    """
    Tests if the LLM correctly parses and classifies various legal intents.
    """
    logger.info(f"--- Testing Query: {query} ---")
    
    # Initialize mock state
    state = AgentState(
        query=query,
        session_id="test_session",
        chat_history=[],
        memory_summary="",
        documents=[],
        case_laws=[],
        generation="",
        iteration_count=0
    )
    
    # Execute Node
    result = qualifier_node(state)
    
    # Check outputs
    assert "law_domain" in result
    assert result["law_domain"] == expected_domain, f"Expected {expected_domain}, got {result.get('law_domain')}"
    
    assert "is_scenario" in result
    assert result["is_scenario"] == expected_scenario, f"Expected scenario={expected_scenario}, got {result.get('is_scenario')}"
    
    assert "requires_case_law" in result
    assert result["requires_case_law"] == expected_case_law, f"Expected case_law={expected_case_law}, got {result.get('requires_case_law')}"
