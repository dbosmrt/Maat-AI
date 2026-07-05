import logging
import traceback
from langchain_core.output_parsers import StrOutputParser
from agent.state import AgentState
from agent.model import ChatModels
from agent.prompt.generator_prompt import get_generator_prompt
from agent.utils.logger import log_node_event, log_system_error

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generator_node(state: AgentState) -> dict:
    """
    Synthesizes the final answer using the original query, the memory summary,
    the retrieved internal documents (laws), and any external case laws (web search).
    """
    query = state.get("query", "")
    documents = state.get("documents", [])
    case_laws = state.get("case_laws", [])
    memory_summary = state.get("memory_summary", "")
    is_scenario = state.get("is_scenario", False)
    
    logger.info("Generator Node: Formulating final response...")
    
    # Format contexts
    docs_text = "\n\n".join(documents) if documents else "No direct internal legal statutes retrieved."
    cases_text = "\n\n".join(case_laws) if case_laws else "No external case laws retrieved."
    memory_text = memory_summary if memory_summary else "No prior conversation history."
    
    # Initialize LLM and Parser
    llm = ChatModels.get_nemotron3super()
    parser = StrOutputParser()
    
    # Customize instructions based on whether it is a hypothetical scenario or direct question
    scenario_instruction = ""
    if is_scenario:
        scenario_instruction = "\n- The user is asking about a specific scenario or event. Apply the laws directly to the people/events mentioned in the query. Provide actionable legal steps if applicable."
    
    prompt = get_generator_prompt(scenario_instruction)
    
    chain = prompt | llm | parser
    
    try:
        generation = chain.invoke({
            "query": query,
            "memory_text": memory_text,
            "docs_text": docs_text,
            "cases_text": cases_text
        })
        
        logger.info("Generator Node: Response successfully generated.")
        log_node_event("generator_node", "SUCCESS")
        
        return {
            "generation": generation
        }
        
    except Exception as e:
        logger.error(f"Generator node failed: {e}")
        log_system_error(traceback.format_exc())
        log_node_event("generator_node", "FAILURE", error_payload=str(e))
        return {"generation": "I apologize, but I encountered an internal error while generating your legal response. Please try again."}
