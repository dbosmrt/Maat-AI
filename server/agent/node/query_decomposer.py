"""Query decomposer node: splits the raw query into retrieval-friendly parts."""

import traceback

from ..model import ChatModels
from ..prompt.query_decomposer_prompt import get_query_decomposer_prompt
from ..state import AgentState, DecomposedQuery
from ..utils.logger import get_logger, log_node_event, log_system_error
from .base import PromptNode

logger = get_logger(__name__)


class QueryDecomposerNode(PromptNode):
    """Node that pre-processes the raw user query into optimized search strings."""

    def __init__(self) -> None:
        super().__init__(name="query_decomposer")

    def execute(self, state: AgentState) -> dict:
        """Pre-process the raw user query into optimized search strings.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary containing `decomposed_query` structure or empty dict.
        """
        query = self._get_query(state)

        logger.info("Decomposing query for hybrid retrieval...")

        llm = ChatModels.get_sarvam_m()
        structured_llm = llm.with_structured_output(DecomposedQuery)
        prompt = self.prompt_template
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


# Singleton instance for backward compatibility
_query_decomposer_node = QueryDecomposerNode()


def query_decomposer_node(state: AgentState) -> dict:
    """Backward-compatible function wrapper."""
    return _query_decomposer_node(state)
