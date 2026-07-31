"""Chat graph construction for the Legal RAG Chatbot.

Builds and compiles the Self-Reflective RAG LangGraph for chat.
Uses node factories for dependency injection and testability.
"""

from langgraph.graph import END, START, StateGraph

from .node.base import BaseNode
from .node.query_decomposer import query_decomposer_node
from .node.qualifier import qualifier_node
from .node.retriever import retriever_node
from .node.reranker import reranker_node
from .node.grader import grader_node
from .node.web_search import web_search_node
from .node.generator import generator_node
from .node.rewriter import rewriter_node
from .state import AgentState
from .utils.logger import get_logger

logger = get_logger(__name__)


def _build_conditional_edges():
    """Define conditional edge logic for the graph."""

    def grader_conditional(state: AgentState) -> str:
        if state.get("retry_retrieval", False):
            return "rewriter"
        if state.get("search_required", False):
            return "web_search"
        return "generator"

    def qualifier_conditional(state: AgentState) -> str:
        if state.get("is_general_chat", False):
            logger.info(
                "General Chat detected! Bypassing retrieval and routing "
                "directly to generator."
            )
            return "generator"
        return "retriever"

    return grader_conditional, qualifier_conditional


def build_chat_graph() -> StateGraph:
    """
    Construct and compile the StateGraph that orchestrates the chat pipeline.

    Returns:
        A compiled `CompiledStateGraph` ready to be invoked with an AgentState.
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)

    # Get conditional edge functions
    grader_conditional, qualifier_conditional = _build_conditional_edges()

    # Add all nodes to the graph (using function wrappers for backward compat)
    workflow.add_node("query_decomposer", query_decomposer_node)
    workflow.add_node("qualifier", qualifier_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("rewriter", rewriter_node)

    # Build the execution sequence
    workflow.add_edge(START, "query_decomposer")
    workflow.add_edge("query_decomposer", "qualifier")

    # Conditional edge from qualifier
    workflow.add_conditional_edges(
        "qualifier",
        qualifier_conditional,
        {
            "generator": "generator",
            "retriever": "retriever",
        },
    )

    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "grader")

    # Conditional edge from grader
    workflow.add_conditional_edges(
        "grader",
        grader_conditional,
        {
            "rewriter": "rewriter",
            "web_search": "web_search",
            "generator": "generator",
        },
    )

    # Retry loop
    workflow.add_edge("rewriter", "retriever")

    workflow.add_edge("web_search", "generator")
    workflow.add_edge("generator", END)

    # Compile the graph
    logger.info("Compiling Chat Graph...")
    return workflow.compile()


# Module-level singleton (module globals are acceptable for singleton patterns)
_COMPILED_GRAPH = None


def get_chat_graph() -> StateGraph:
    """Get or create the compiled chat graph (singleton pattern).

    Returns:
        Compiled StateGraph instance.
    """
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_chat_graph()
    return _COMPILED_GRAPH
