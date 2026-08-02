"""Generator node: synthesizes the final legal response."""

import traceback
from typing import Union

from ..model import ChatModels
from ..prompt.generator_prompt import get_generator_prompt
from ..state import AgentState, GeneratorOutput
from ..utils.logger import get_logger, log_node_event, log_system_error
from .base import PromptNode

logger = get_logger(__name__)


class GeneratorNode(PromptNode):
    """Node that synthesizes the final legal response using retrieved contexts."""

    def __init__(self) -> None:
        super().__init__(name="generator")

    def execute(self, state: AgentState) -> dict:
        """Synthesize the final answer using retrieved contexts and memory.

        Args:
            state: Current LangGraph state. Reads `query`, `documents`,
                `case_laws`, `memory_summary`, and `is_scenario`.

        Returns:
            Dictionary with `generation` and `law_domain` fields.
            On LLM failure, returns a graceful fallback message.
        """
        query = state.get("query", "")
        documents = state.get("documents", [])
        case_laws = state.get("case_laws", [])
        memory_summary = state.get("memory_summary", "")
        is_scenario = state.get("is_scenario", False)

        # Get user model settings from state (injected by API route)
        user_chat_model = state.get("user_chat_model")
        user_temperature = state.get("user_temperature")
        user_max_tokens = state.get("user_max_tokens")
        user_top_p = state.get("user_top_p")

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

        # Initialize LLM with user settings or defaults
        llm = ChatModels.get_from_user_settings(
            preferred_model=user_chat_model,
            temperature=user_temperature,
            top_p=user_top_p,
            max_tokens=user_max_tokens,
        )
        structured_llm = llm.with_structured_output(GeneratorOutput)

        # Customize instructions based on whether it is a scenario or direct question
        scenario_instruction = ""
        if is_scenario:
            scenario_instruction = (
                "\n- The user is asking about a specific scenario or event. "
                "Apply the laws directly to the people/events mentioned in the "
                "query. Provide actionable legal steps if applicable."
            )

        prompt = get_generator_prompt(scenario_instruction)

        chain = prompt | structured_llm

        try:
            result: Union[dict, GeneratorOutput] = chain.invoke(
                {
                    "query": query,
                    "memory_text": memory_text,
                    "docs_text": docs_text,
                    "cases_text": cases_text,
                }
            )

            logger.info("Generator Node: Response successfully generated.")
            log_node_event("generator_node", "SUCCESS")

            # Type guard: result could be dict or GeneratorOutput depending on LLM implementation
            if isinstance(result, dict):
                generation = result.get("generation", "")
                law_domain = result.get("law_domain", "General")
            else:
                generation = result.generation
                law_domain = result.law_domain

            return {
                "generation": generation,
                "law_domain": law_domain,
            }

        except (RuntimeError, ValueError, ConnectionError) as exc:
            logger.error("Generator node failed: %s", exc)
            log_system_error(traceback.format_exc())
            log_node_event("generator_node", "FAILURE", error_payload=str(exc))
            return {
                "generation": (
                    "I apologize, but I encountered an internal error while "
                    "generating your legal response. Please try again."
                ),
                "law_domain": "General",
            }


# Singleton instance for backward compatibility
_generator_node = GeneratorNode()


def generator_node(state: AgentState) -> dict:
    """Backward-compatible function wrapper."""
    return _generator_node(state)
