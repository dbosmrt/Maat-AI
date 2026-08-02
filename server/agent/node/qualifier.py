"""Qualifier node: classifies the user query into a domain and intent."""

from ..model import ChatModels
from ..prompt.qualifier_prompt import get_qualifier_prompt
from ..state import AgentState, QueryClassification
from ..utils.logger import get_logger
from .base import PromptNode

logger = get_logger(__name__)


class QualifierNode(PromptNode):
    """Node that classifies the user's query intent and legal domain."""

    def __init__(self) -> None:
        super().__init__(name="qualifier", default_model="sarvam_m")
        # Override prompt getter to use the correct module
        self._prompt_template = get_qualifier_prompt()

    def execute(self, state: AgentState) -> dict:
        """Analyze the user's query to understand its intent and domain.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary with law_domain, is_scenario, requires_case_law, is_general_chat.
        """
        query = self._get_query(state)

        logger.info("Qualifying query: %s", query)

        llm = self._get_llm(state)
        structured_llm = llm.with_structured_output(QueryClassification)

        chain = self.prompt_template | structured_llm

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

        return {
            "law_domain": classification_dict.get("law_domain", "General"),
            "is_scenario": classification_dict.get("is_scenario", False),
            "requires_case_law": classification_dict.get("requires_case_law", False),
            "is_general_chat": classification_dict.get("is_general_chat", False),
        }


# Singleton instance for backward compatibility
_qualifier_node = QualifierNode()


def qualifier_node(state: AgentState) -> dict:
    """Backward-compatible function wrapper."""
    return _qualifier_node(state)
