import logging
from duckduckgo_search import DDGS
from langchain_core.output_parsers import JsonOutputParser
from agent.state import AgentState, SearchQueries
from agent.model import ChatModels
from agent.prompt.search_query_prompt import get_search_query_prompt
from agent.utils.logger import log_node_event, log_system_error
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum raw query length before we invoke the LLM summarizer
QUERY_SUMMARIZE_THRESHOLD = 120


def _summarize_query(query: str) -> list[str]:
    """
    Uses the LLM to extract 2-3 concise, focused search terms from a long query.
    Falls back to a truncated version of the raw query if the LLM fails.
    """
    try:
        llm = ChatModels.get_nemotron3super()
        structured_llm = llm.with_structured_output(SearchQueries)
        prompt = get_search_query_prompt()
        
        chain = prompt | structured_llm
        result = chain.invoke({
            "query": query,
            "format_instructions": "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
        })
        
        result_dict = result.dict() if hasattr(result, "dict") else dict(result)
        queries = result_dict.get("search_queries", [])
        
        if queries:
            logger.info(f"Search Summarizer generated {len(queries)} queries: {queries}")
            log_node_event("web_search_summarizer", "SUCCESS")
            return queries
    except Exception as e:
        logger.warning(f"Search query summarizer failed: {e}. Using truncated query.")
        log_system_error(traceback.format_exc())
        log_node_event("web_search_summarizer", "PARSING_RETRY", error_payload=str(e))
    
    # Fallback: just truncate the raw query
    return [query[:QUERY_SUMMARIZE_THRESHOLD]]


def web_search_node(state: AgentState) -> dict:
    """
    Executes a web search for the query, particularly targeted at finding 
    Indian case laws, judicial precedents, and legal articles.
    For long queries, uses an LLM agent to first extract concise search terms.
    """
    query = state.get("query", "")
    
    if not query:
        return {"case_laws": []}
    
    # Decide whether to summarize the query or use it directly
    if len(query) > QUERY_SUMMARIZE_THRESHOLD:
        logger.info(f"Query is {len(query)} chars, invoking search summarizer agent...")
        search_queries = _summarize_query(query)
    else:
        # Short query: enhance it directly
        suffix = "India Supreme Court High Court judgments case law" if state.get("requires_case_law", False) else "Indian law"
        search_queries = [f"{query} {suffix}"]
    
    logger.info(f"Web Search Node: Executing {len(search_queries)} searches...")
    
    try:
        all_results = []
        seen_links = set()
        
        with DDGS() as ddgs:
            for sq in search_queries:
                logger.info(f"  Searching: '{sq}'")
                ddg_results = ddgs.text(sq, region='in-en', max_results=3)
                
                for r in ddg_results:
                    link = r.get("href", "")
                    # Deduplicate by URL
                    if link in seen_links:
                        continue
                    seen_links.add(link)
                    
                    title = r.get("title", "")
                    body = r.get("body", "")
                    formatted_result = f"[External Source: {title}] ({link})\n{body}\n"
                    all_results.append(formatted_result)
                    
        logger.info(f"Web Search Node: Found {len(all_results)} unique external results.")
        
        log_node_event("web_search_node", "SUCCESS")
        
        return {
            "case_laws": all_results
        }
    except Exception as e:
        logger.error(f"Web Search Node failed: {e}")
        log_system_error(traceback.format_exc())
        log_node_event("web_search_node", "FAILURE", error_payload=str(e))
        return {"case_laws": []}
