"""Web Search node: searches for case law and legal articles."""

import os
import traceback
from typing import List

from ..utils.logger import get_logger, log_node_event, log_system_error
from duckduckgo_search import DDGS
from ..state import AgentState, SearchQueries
from ..model import ChatModels
from ..prompt.search_query_prompt import get_search_query_prompt
from .base import BaseNode

logger = get_logger(__name__)

# Configuration from environment with defaults
DDGS_REGION = os.environ.get("DDGS_REGION", "in-en")
DDGS_MAX_RESULTS = int(os.environ.get("DDGS_MAX_RESULTS", "3"))

# Maximum raw query length before we invoke the LLM summarizer
QUERY_SUMMARIZE_THRESHOLD = int(os.environ.get("QUERY_SUMMARIZE_THRESHOLD", "120"))


class WebSearchNode(BaseNode):
    """Node that executes web search for legal case law and articles."""

    def __init__(self) -> None:
        super().__init__(name="web_search")

    def execute(self, state: AgentState) -> dict:
        """Execute web search for the query, targeting Indian case laws.

        Args:
            state: Current LangGraph state.

        Returns:
            Dictionary with `case_laws` list.
        """
        query = state.get("query", "")

        if not query:
            return {"case_laws": []}

        # Decide whether to summarize the query or use it directly
        if len(query) > QUERY_SUMMARIZE_THRESHOLD:
            logger.info(
                "Query is %d chars, invoking search summarizer agent...",
                len(query),
            )
            search_queries = self._summarize_query(query)
        else:
            # Short query: enhance it directly
            suffix = (
                "India Supreme Court High Court judgments case law"
                if state.get("requires_case_law", False)
                else "Indian law"
            )
            search_queries = [f"{query} {suffix}"]

        logger.info("Web Search Node: Executing %d searches...", len(search_queries))

        try:
            all_results = []
            seen_links = set()

            with DDGS() as ddgs:
                for sq in search_queries:
                    logger.info("  Searching: '%s'", sq)
                    ddg_results = ddgs.text(
                        sq, region=DDGS_REGION, max_results=DDGS_MAX_RESULTS
                    )

                    for r in ddg_results:
                        link = r.get("href", "")
                        # Deduplicate by URL
                        if link in seen_links:
                            continue
                        seen_links.add(link)

                        title = r.get("title", "")
                        body = r.get("body", "")
                        formatted_result = (
                            f"[External Source: {title}] ({link})\n{body}\n"
                        )
                        all_results.append(formatted_result)

            logger.info(
                "Web Search Node: Found %d unique external results.",
                len(all_results),
            )

            log_node_event("web_search_node", "SUCCESS")

            return {"case_laws": all_results}
        except Exception as exc:
            logger.error("Web Search Node failed: %s", exc)
            log_system_error(traceback.format_exc())
            log_node_event("web_search_node", "FAILURE", error_payload=str(exc))
            return {"case_laws": []}

    def _summarize_query(self, query: str) -> List[str]:
        """Use LLM to extract focused search terms from a long query.

        Args:
            query: The long user query.

        Returns:
            List of focused search queries.
        """
        try:
            llm = ChatModels.get_sarvam_m()
            structured_llm = llm.with_structured_output(SearchQueries)
            prompt = get_search_query_prompt()

            chain = prompt | structured_llm
            result = chain.invoke(
                {
                    "query": query,
                    "format_instructions": (
                        "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
                    ),
                }
            )

            result_dict = (
                result.model_dump()
                if hasattr(result, "model_dump")
                else (result.dict() if hasattr(result, "dict") else dict(result))
            )
            queries = result_dict.get("search_queries", [])

            if queries:
                logger.info(
                    "Search Summarizer generated %d queries: %s",
                    len(queries),
                    queries,
                )
                log_node_event("web_search_summarizer", "SUCCESS")
                return queries
        except Exception as exc:
            logger.warning(
                "Search query summarizer failed: %s. Using truncated query.", exc
            )
            log_system_error(traceback.format_exc())
            log_node_event("web_search_summarizer", "PARSING_RETRY", error_payload=str(exc))

        # Fallback: just truncate the raw query
        return [query[:QUERY_SUMMARIZE_THRESHOLD]]


# Singleton instance for backward compatibility
_web_search_node = WebSearchNode()


def web_search_node(state: AgentState) -> dict:
    """Backward-compatible function wrapper."""
    return _web_search_node(state)

