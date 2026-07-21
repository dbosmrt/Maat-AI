import pytest
import logging
from agent.utils.logger import get_logger
from agent.state import AgentState
from agent.node.grader import grader_node


logger = get_logger(__name__)

def test_grader_node_relevant():
    """Test Grader with relevant documents. Should bypass web search."""
    state = AgentState(
        query="What is the punishment for theft?",
        documents=["[Source: BNS] The punishment for theft is up to 3 years."],
        requires_case_law=False,
        session_id="test", chat_history=[], memory_summary="", case_laws=[], generation="", iteration_count=0
    )
    result = grader_node(state)
    assert result.get("search_required") is False

def test_grader_node_irrelevant():
    """Test Grader with irrelevant documents. Should route to web search after retries exhausted."""
    state = AgentState(
        query="What is the punishment for theft?",
        documents=["[Source: BNS] A person who commits murder shall be punished with death."],
        requires_case_law=False,
        session_id="test", chat_history=[], memory_summary="", case_laws=[], generation="",
        iteration_count=3  # Exhaust retries (max 2) to force web search
    )
    result = grader_node(state)
    assert result.get("search_required") is True

def test_grader_node_case_law_request():
    """Test Grader when user explicitly asks for case law. Should immediately route to web search."""
    state = AgentState(
        query="Give me a supreme court case about theft.",
        documents=["[Source: BNS] The punishment for theft is up to 3 years."],
        requires_case_law=True,
        session_id="test", chat_history=[], memory_summary="", case_laws=[], generation="", iteration_count=0
    )
    result = grader_node(state)
    assert result.get("search_required") is True
