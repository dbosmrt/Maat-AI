"""Generator node: synthesizes the final legal response."""

import traceback

from langchain_core.output_parsers import StrOutputParser

from agent.model import ChatModels
from agent.prompt.generator_prompt import get_generator_prompt
from agent.state import AgentState
from agent.utils.logger import get_logger, log_node_event, log_system_error

logger = get_logger(__name__)


def generator_node(state: AgentState) -> dict:
    """
    Synthesize the final answer using retrieved contexts and memory.

    Args:
        state: Current LangGraph state. Reads `query`, `documents`,
            `case_laws`, `memory_summary`, and `is_scenario`.

    Returns:
        A dict with the new `generation` field. On LLM failure, returns a
        graceful fallback message.
    """
    query = state.get("query", "")
    documents = state.get("documents", [])
    case_laws = state.get("case_laws", [])
    memory_summary = state.get("memory_summary", "")
    is_scenario = state.get("is_scenario", False)

    logger.info("Generator Node: Formulating final response...")

    # Format contexts
    docs_text = (
        "\n\n".join(documents)
        if documents
        else "No direct internal legal statutes retrieved."
    )
    cases_text = (
        "\n\n".join(case_laws)
        if case_laws
        else "No external case laws retrieved."
    )
    memory_text = (
        memory_summary if memory_summary else "No prior conversation history."
    )

    # Initialize LLM and Parser
    llm = ChatModels.get_sarvam_m()
    parser = StrOutputParser()

    # Customize instructions based on whether it is a scenario or direct question
    scenario_instruction = ""
    if is_scenario:
        scenario_instruction = (
            "\n- The user is asking about a specific scenario or event. "
            "Apply the laws directly to the people/events mentioned in the "
            "query. Provide actionable legal steps if applicable."
        )

    prompt = get_generator_prompt(scenario_instruction)

    chain = prompt | llm | parser

    try:
        generation = chain.invoke(
            {
                "query": query,
                "memory_text": memory_text,
                "docs_text": docs_text,
                "cases_text": cases_text,
            }
        )

        logger.info("Generator Node: Response successfully generated.")
        log_node_event("generator_node", "SUCCESS")

        return {"generation": generation}

    except (RuntimeError, ValueError, ConnectionError) as exc:
        logger.error("Generator node failed: %s", exc)
        log_system_error(traceback.format_exc())
        log_node_event("generator_node", "FAILURE", error_payload=str(exc))
        return {
            "generation": (
                "I apologize, but I encountered an internal error while "
                "generating your legal response. Please try again."
            )
        }
