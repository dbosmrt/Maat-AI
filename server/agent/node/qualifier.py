from agent.utils.logger import get_logger, log_node_event, log_system_error
from langchain_core.output_parsers import JsonOutputParser
from agent.state import AgentState, QueryClassification
from agent.model import ChatModels
from agent.prompt.qualifier_prompt import get_qualifier_prompt
import traceback

logger = get_logger(__name__)

def qualifier_node(state: AgentState) -> dict:
    """
    Analyzes the user's query to understand the intent, domain, and if external search is needed.
    """
    query = state.get("query", "")
    
    if not query:
        logger.warning("qualifier_node: No query found.")
        return {}
        
    logger.info(f"Qualifying query: {query}")
    
    # Initialize the LLM (NVIDIA Nemotron)
    llm = ChatModels.get_sarvam_m()
    
    # Bind the Pydantic schema using structured output
    structured_llm = llm.with_structured_output(QueryClassification)
    prompt = get_qualifier_prompt()
    chain = prompt | structured_llm
    
    try:
        classification = chain.invoke({
            "query": query,
            "format_instructions": "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
        })
        
        # Pydantic object
        classification_dict = classification.model_dump() if hasattr(classification, "dict") else dict(classification)
        
        logger.info(
            f"Query Qualified -> Domain: {classification_dict.get('law_domain')} | "
            f"Scenario: {classification_dict.get('is_scenario')} | "
            f"Needs Case Law: {classification_dict.get('requires_case_law')} | "
            f"General Chat: {classification_dict.get('is_general_chat')}"
        )
        
        log_node_event("qualifier_node", "SUCCESS")
        
        return {
            "law_domain": classification_dict.get("law_domain", "General"),
            "is_scenario": classification_dict.get("is_scenario", False),
            "requires_case_law": classification_dict.get("requires_case_law", False),
            "is_general_chat": classification_dict.get("is_general_chat", False)
        }
    except Exception as e:
        logger.error(f"Qualifier node failed: {e}")
        log_system_error(traceback.format_exc())
        log_node_event("qualifier_node", "PARSING_RETRY", error_payload=str(e))
        # Safe fallback values if the LLM errors out
        return {
            "law_domain": "General",
            "is_scenario": False,
            "requires_case_law": False,
            "is_general_chat": False
        }
