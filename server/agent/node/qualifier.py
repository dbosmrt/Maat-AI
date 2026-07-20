"""Qualifier node: classifies the user query into a domain and intent."""

import traceback

from agent.model import ChatModels
from agent.prompt.qualifier_prompt import get_qualifier_prompt
from agent.state import AgentState, QueryClassification
from agent.utils.logger import (
    get_logger,
    log_node_event,
    log_system_error,
)

logger = get_logger(__name__)


def qualifier_node(state: AgentState) -> dict:
    """
    Analyze the user's query to understand its intent, domain, and whether
    external search is needed.

    Args:
        state: Current LangGraph state.

    Returns:
        A dict containing `law_domain`, `is_scenario`, `requires_case_law`,
        and `is_general_chat`. Empty dict if no query.
    """
    query = state.get("query", "")

    if not query:
        logger.warning("qualifier_node: No query found.")
        return {}

    logger.info("Qualifying query: %s", query)

    llm = ChatModels.get_sarvam_m()
    structured_llm = llm.with_structured_output(QueryClassification)
    prompt = get_qualifier_prompt()
    chain = prompt | structured_llm

    try:
        classification = chain.invoke(
            {
                "query": query,
                "format_instructions": (
                    "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
                ),
            }
        )

        classification_dict = (
            classification.model_dump()
            if hasattr(classification, "model_dump")
            else dict(classification)
        )

        logger.info(
            "Query Qualified -> Domain: %s | Scenario: %s | "
            "Needs Case Law: %s | General Chat: %s",
            classification_dict.get("law_domain"),
            classification_dict.get("is_scenario"),
            classification_dict.get("requires_case_law"),
            classification_dict.get("is_general_chat"),
        )

        log_node_event("qualifier_node", "SUCCESS")

        return {
            "law_domain": classification_dict.get("law_domain", "General"),
            "is_scenario": classification_dict.get("is_scenario", False),
            "requires_case_law": classification_dict.get(
                "requires_case_law", False
            ),
            "is_general_chat": classification_dict.get("is_general_chat", False),
        }
    except (RuntimeError, ValueError, ConnectionError) as exc:
        logger.error("Qualifier node failed: %s", exc)
        log_system_error(traceback.format_exc())
        log_node_event("qualifier_node", "PARSING_RETRY", error_payload=str(exc))
        # Safe fallback values if the LLM errors out
        return {
            "law_domain": "General",
            "is_scenario": False,
            "requires_case_law": False,
            "is_general_chat": False,
        }
