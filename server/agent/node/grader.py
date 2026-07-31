"""Grader node: evaluates whether the retrieved documents answer the query."""

import os
import traceback
from typing import Optional

from ..model import ChatModels
from ..prompt.grader_prompt import get_grader_prompt
from ..state import AgentState, DocumentGrade
from ..utils.grader_utils import calculate_max_retries
from ..utils.logger import (
    LOG_DIR,
    get_logger,
    log_node_event,
    log_system_error,
)
from .base import BaseNode

logger = get_logger(__name__)

# Minimum score to accept LLM's is_relevant=True verdict.
# Below this, we override to False regardless of LLM output.
MIN_RELEVANCE_SCORE_OVERRIDE = 0.5


class GraderNode(BaseNode):
    """Node that evaluates if retrieved documents are sufficient to answer the query."""

    def __init__(self) -> None:
        super().__init__(name="grader")

    def execute(self, state: AgentState) -> dict:
        """Evaluate if the retrieved documents are sufficient to answer the query.

        Routes between the rewriter, web search, and generator paths based on
        the LLM-assessed relevance and remaining retry budget.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary with `search_required` and `retry_retrieval` flags.
        """
        query = state.get("query", "")
        documents = state.get("documents", [])
        requires_case_law = state.get("requires_case_law", False)

        if not query:
            logger.warning("grader_node: Missing query.")
            return {"search_required": False, "retry_retrieval": False}

        # Heuristic 1: If user wants Case Law, our DB (statutes only) won't have it.
        if requires_case_law:
            logger.info(
                "Grader Node: Query requires external case law. Routing to Web Search."
            )
            return {"search_required": True, "retry_retrieval": False}

        # Heuristic 2: If no documents survived the re-ranker, we must search.
        if not documents:
            return self._handle_no_documents(state)

        logger.info("Grader Node: Evaluating document relevance...")

        llm = ChatModels.get_sarvam_m()
        structured_llm = llm.with_structured_output(DocumentGrade)

        docs_text = "\n".join(documents)

        prompt = get_grader_prompt()
        chain = prompt | structured_llm

        try:
            grade = chain.invoke(
                {
                    "query": query,
                    "docs_text": docs_text,
                    "format_instructions": (
                        "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
                    ),
                }
            )

            grade_dict = (
                grade.model_dump() if hasattr(grade, "model_dump") else dict(grade)
            )
            is_relevant = grade_dict.get("is_relevant", False)
            relevance_score = grade_dict.get("context_relevance_score", 1.0)
            diversity_analysis = grade_dict.get("chunk_diversity", "")
            failure_reason = grade_dict.get("failure_reason")

            logger.info(
                "Grader Evaluation -> Relevance: %s | Score: %s | Diversity: %s",
                is_relevant,
                relevance_score,
                diversity_analysis,
            )
            if failure_reason:
                logger.info("Grader Failure Reason: %s", failure_reason)

            # Code-level safety override: if the LLM said relevant but score is
            # below our minimum threshold, override to irrelevant.
            if is_relevant and relevance_score < MIN_RELEVANCE_SCORE_OVERRIDE:
                logger.warning(
                    "Grader Override: LLM said is_relevant=True but score=%s "
                    "< %s. Overriding to is_relevant=False.",
                    relevance_score,
                    MIN_RELEVANCE_SCORE_OVERRIDE,
                )
                is_relevant = False

            # Health Check Auditing
            if relevance_score < 0.4:
                warning_msg = (
                    f"[{query}] Critical Context Starvation! "
                    f"Score: {relevance_score}. Diversity: {diversity_analysis}"
                )
                logger.warning(warning_msg)
                health_log = os.path.join(LOG_DIR, "retrieval_health_warnings.log")
                with open(health_log, "a", encoding="utf-8") as f:
                    f.write(warning_msg + "\n")

            if is_relevant:
                logger.info(
                    "Grader Node: Documents are highly relevant. Bypassing Web Search."
                )
                log_node_event("grader_node", "SUCCESS")
                return {"search_required": False, "retry_retrieval": False}
            return self._retry_or_search(state)

        except (RuntimeError, ValueError, ConnectionError) as exc:
            logger.error("Grader node failed: %s", exc)
            log_system_error(traceback.format_exc())
            log_node_event(
                "grader_node", "PARSING_RETRY", error_payload=str(exc)
            )
            # Safe fallback: try generating with what we have
            return {"search_required": False, "retry_retrieval": False}

    def _handle_no_documents(self, state: AgentState) -> dict:
        """Handle case where no documents were retrieved.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary with routing flags.
        """
        iteration_count = state.get("iteration_count", 0)
        max_retries = calculate_max_retries(state.get("decomposed_query", {}))
        if iteration_count < max_retries:
            logger.info(
                "Grader Node: No documents provided (Iteration %d). "
                "Routing to Rewriter.",
                iteration_count,
            )
            return {"search_required": False, "retry_retrieval": True}
        logger.info(
            "Grader Node: No documents provided. Max retries reached. "
            "Routing to Web Search."
        )
        return {"search_required": True, "retry_retrieval": False}

    def _retry_or_search(self, state: AgentState) -> dict:
        """Return the next-step signal based on iteration count.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary with routing flags.
        """
        iteration_count = state.get("iteration_count", 0)
        max_retries = calculate_max_retries(state.get("decomposed_query", {}))
        if iteration_count < max_retries:
            logger.info(
                "Grader Node: Documents are irrelevant (Iteration %d). "
                "Retrying retrieval loop.",
                iteration_count,
            )
            log_node_event("grader_node", "RETRY")
            return {"search_required": False, "retry_retrieval": True}
        logger.info(
            "Grader Node: Documents are irrelevant and max retries reached. "
            "Routing to Web Search."
        )
        log_node_event("grader_node", "SUCCESS")
        return {"search_required": True, "retry_retrieval": False}


# Singleton instance for backward compatibility
_grader_node = GraderNode()


def grader_node(state: AgentState) -> dict:
    """Backward-compatible function wrapper."""
    return _grader_node(state)

