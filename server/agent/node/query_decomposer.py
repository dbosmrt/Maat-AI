"""Query decomposer node: splits the raw query into retrieval-friendly parts."""

import traceback

from agent.model import ChatModels
from agent.prompt.query_decomposer_prompt import get_query_decomposer_prompt
from agent.state import AgentState, DecomposedQuery
from agent.utils.logger import (
    get_logger,
    log_node_event,
    log_system_error,
)

logger = get_logger(__name__)


def query_decomposer_node(state: AgentState) -> dict:
    """
    Pre-process the raw user query into optimized search strings
    (semantic, statutory, procedural) for the hybrid RAG engine.

    Args:
        state: Current LangGraph state.

    Returns:
        A dict containing `decomposed_query` (the parsed structure) or an
        empty dict if the query is missing.
    """
    query = state.get("query", "")

    if not query:
        logger.warning("query_decomposer_node: No query found.")
        return {}

    logger.info("Decomposing query for hybrid retrieval...")

    llm = ChatModels.get_sarvam_m()
    structured_llm = llm.with_structured_output(DecomposedQuery)
    prompt = get_query_decomposer_prompt()
    chain = prompt | structured_llm

    try:
        decomposed = chain.invoke(
            {
                "query": query,
                "format_instructions": (
                    "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
                ),
            }
        )

        decomposed_dict = (
            decomposed.model_dump()
            if hasattr(decomposed, "model_dump")
            else dict(decomposed)
        )

        logger.info(
            "Query Decomposed -> Domain: %s | Statutory Focus: %s",
            decomposed_dict.get("domain"),
            decomposed_dict.get("statutory_focus"),
        )

        log_node_event("query_decomposer_node", "SUCCESS")

        return {"decomposed_query": decomposed_dict}

    except (RuntimeError, ValueError, ConnectionError) as exc:
        logger.error("Query decomposer failed: %s", exc)
        log_system_error(traceback.format_exc())
        log_node_event(
            "query_decomposer_node", "PARSING_RETRY", error_payload=str(exc)
        )

        # Fallback to basic decomposition
        return {
            "decomposed_query": {
                "semantic_focus": query,
                "statutory_focus": "",
                "procedural_focus": "",
                "domain": "criminal",
            }
        }
