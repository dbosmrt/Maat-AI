import pytest
import logging
from agent.utils.logger import get_logger
from agent.state import AgentState
from agent.node.reranker import reranker_node


logger = get_logger(__name__)

def test_reranker_node():
    """
    Test the Re-ranker Node's ability to filter out irrelevant chunks.
    """
    query = "What is the punishment for theft?"
    
    # Mocking 4 documents. Only index 1 and 3 are actually relevant to theft.
    # Index 0 and 2 are noise/irrelevant.
    mock_documents = [
        "[Source: BSA] This section talks about property rights in a divorce.",
        "[Source: BNS] The punishment for theft is imprisonment which may extend to 3 years.",
        "[Source: BNS] A person who commits murder shall be punished with death or life imprisonment.",
        "[Source: BNSS] Whoever commits theft in a dwelling house is subject to strict penalties under section 380."
    ]
    
    logger.info("--- Testing Re-ranker Node ---")
    
    # Mock state
    state = AgentState(
        query=query,
        session_id="test",
        chat_history=[],
        memory_summary="",
        documents=mock_documents,
        case_laws=[],
        generation="",
        iteration_count=0
    )
    
    # Execute node
    result = reranker_node(state)
    
    assert "documents" in result
    filtered_docs = result["documents"]
    
    # We expect it to filter out index 0 and 2, leaving 2 documents
    assert len(filtered_docs) <= 3, "It should have filtered out at least one irrelevant document."
    
    # Verify that the theft documents survived
    survived_text = " ".join(filtered_docs).lower()
    assert "theft" in survived_text, "The relevant theft documents were incorrectly filtered out!"
    assert "divorce" not in survived_text, "The irrelevant divorce document was not filtered out!"
    
    logger.info(f"Re-ranker test passed! Kept docs: {len(filtered_docs)}")
