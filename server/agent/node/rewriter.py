"""Rewriter node: rewrites the user query to optimize for better vector search results."""

import traceback

from ..model import ChatModels
from ..prompt.rewriter_prompt import get_rewriter_prompt
from ..state import AgentState, RewriterOutput
from ..utils.logger import get_logger, log_node_event, log_system_error
from .base import PromptNode

logger = get_logger(__name__)


class RewriterNode(PromptNode):
    """Node that rewrites queries for better retrieval on retry attempts."""

    def __init__(self) -> None:
        super().__init__(name="rewriter")

    def execute(self, state: AgentState) -> dict:
        """Rewrite the user query to optimize for better vector search results.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary with `query` (rewritten) and incremented `iteration_count`.
        """
        query = state.get("query", "")
        iteration_count = state.get("iteration_count", 0)

        logger.info(
            "Rewriting query for better retrieval (Iteration %d). Original: %s",
            iteration_count + 1,
            query,
        )

        llm = ChatModels.get_sarvam_m()
        # Pass iteration count to get the appropriate prompt strategy
        prompt = get_rewriter_prompt(iteration_count)
        structured_llm = llm.with_structured_output(RewriterOutput)
        chain = prompt | structured_llm

        try:
            result = chain.invoke({"query": query})
            result_typed = result  # Could be dict or RewriterOutput

            # Type guard: result could be dict or RewriterOutput depending on LLM implementation
            if isinstance(result_typed, dict):
                new_query = result_typed.get("rewritten_query", query).strip(' "\'')
            else:
                new_query = result_typed.rewritten_query.strip(' "\'')

            logger.info("Rewritten Query: %s", new_query)
            log_node_event("rewriter_node", "SUCCESS")

            return {
                "query": new_query,
                "iteration_count": iteration_count + 1,
            }
        except Exception as exc:
            logger.error("Rewriter failed: %s", exc)
            log_system_error(traceback.format_exc())
            return {"iteration_count": iteration_count + 1}


# Singleton instance for backward compatibility
_rewriter_node = RewriterNode()


def rewriter_node(state: AgentState) -> dict:
    """Backward-compatible function wrapper."""
    return _rewriter_node(state)
