from agent.utils.logger import get_logger, log_node_event, log_system_error
import traceback
from agent.state import AgentState, DecomposedQuery
from agent.model import ChatModels
from agent.prompt.query_decomposer_prompt import get_query_decomposer_prompt

logger = get_logger(__name__)

def query_decomposer_node(state: AgentState) -> dict:
    """
    Pre-processes the raw user query into optimized search strings
    (semantic, statutory, procedural) for the hybrid RAG engine.
    """
    query = state.get("query", "")
    
    if not query:
        logger.warning("query_decomposer_node: No query found.")
        return {}
        
    logger.info("Decomposing query for hybrid retrieval...")
    
    llm = ChatModels.get_sarvam_m()
    structured_llm = llm.with_structured_output(DecomposedQuery)
    prompt = get_query_decomposer_prompt()
    chain = prompt | structured_llm
    
    try:
        decomposed = chain.invoke({
            "query": query,
            "format_instructions": "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
        })
        
        decomposed_dict = decomposed.dict() if hasattr(decomposed, "dict") else dict(decomposed)
        
        logger.info(f"Query Decomposed -> Domain: {decomposed_dict.get('domain')} | Statutory Focus: {decomposed_dict.get('statutory_focus')}")
        
        log_node_event("query_decomposer_node", "SUCCESS")
        
        return {
            "decomposed_query": decomposed_dict
        }
        
    except Exception as e:
        logger.error(f"Query decomposer failed: {e}")
        log_system_error(traceback.format_exc())
        log_node_event("query_decomposer_node", "PARSING_RETRY", error_payload=str(e))
        
        # Fallback to basic decomposition
        return {
            "decomposed_query": {
                "semantic_focus": query,
                "statutory_focus": "",
                "procedural_focus": "",
                "domain": "criminal"
            }
        }
