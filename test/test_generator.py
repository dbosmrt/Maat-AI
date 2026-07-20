import pytest
import logging
from agent.utils.logger import get_logger
from agent.state import AgentState
from agent.node.generator import generator_node


logger = get_logger(__name__)

def test_generator_node():
    """Test the Generator Node's ability to synthesize a final response from mocked state context."""
    logger.info("--- Testing Generator Node ---")
    
    state = AgentState(
        query="What is the punishment for theft according to the BNS?",
        is_scenario=False,
        documents=["[Source: BNS - CHAPTER XVII - Section 303] Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both."],
        case_laws=[],
        memory_summary="The user previously asked about property crimes.",
        requires_case_law=False,
        session_id="test", chat_history=[], generation="", iteration_count=0
    )
    
    result = generator_node(state)
    
    assert "generation" in result
    generation = result["generation"]
    
    assert isinstance(generation, str)
    assert len(generation) > 50, "Generation output is suspiciously short."
    assert "three years" in generation.lower() or "3 years" in generation.lower(), "Generator failed to include the core fact from the context."
    assert "303" in generation, "Generator failed to cite the section number from the context."
    
    logger.info(f"Final Generation Output:\n\n{generation}")
