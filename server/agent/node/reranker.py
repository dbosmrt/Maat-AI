"""Reranker node: filters retrieved documents for relevance."""

from ..utils.logger import get_logger, log_node_event, log_system_error
from ..state import AgentState, DocumentRanking
from ..model import ChatModels
from ..prompt.reranker_prompt import get_reranker_prompt
from .base import PromptNode

logger = get_logger(__name__)


class RerankerNode(PromptNode):
    """Node that filters retrieved documents to keep only relevant chunks."""

    def __init__(self) -> None:
        super().__init__(name="reranker")

    def execute(self, state: AgentState) -> dict:
        """Filter documents based on relevance to the query.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary with filtered `documents` list.
        """
        query = state.get("query", "")
        documents = state.get("documents", [])

        if not query or not documents:
            logger.warning("reranker_node: Missing query or documents.")
            return {"documents": documents}

        logger.info("Re-ranking %d retrieved documents...", len(documents))

        llm = ChatModels.get_sarvam_m()
        structured_llm = llm.with_structured_output(DocumentRanking)
        prompt = self.prompt_template
        chain = prompt | structured_llm

        # Format the documents for the prompt
        docs_text = ""
        for i, doc in enumerate(documents):
            docs_text += f"\n--- Document Index: {i} ---\n{doc}\n"

        try:
            ranking = chain.invoke(
                {
                    "query": query,
                    "docs_text": docs_text,
                    "format_instructions": "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN.",
                }
            )

            ranking_dict = (
                ranking.model_dump()
                if hasattr(ranking, "model_dump")
                else (ranking.dict() if hasattr(ranking, "dict") else dict(ranking))
            )
            relevant_indices = ranking_dict.get("relevant_indices", [])

            # Filter the original documents list based on the returned indices
            filtered_docs = []
            for idx in relevant_indices:
                if 0 <= idx < len(documents):
                    filtered_docs.append(documents[idx])

            logger.info(
                "Re-ranker kept %d out of %d documents.",
                len(filtered_docs),
                len(documents),
            )

            log_node_event("reranker_node", "SUCCESS")

            return {"documents": filtered_docs}

        except Exception as exc:
            logger.error("Re-ranker node failed: %s", exc)
            log_system_error(traceback.format_exc())
            log_node_event("reranker_node", "PARSING_RETRY", error_payload=str(exc))
            # Fallback: if the LLM fails, just return all documents
            return {"documents": documents}


# Singleton instance for backward compatibility
_reranker_node = RerankerNode()


def reranker_node(state: AgentState) -> dict:
    """Backward-compatible function wrapper."""
    return _reranker_node(state)