from agent.utils.logger import get_logger, log_node_event, log_system_error
from langchain_core.output_parsers import StrOutputParser
from agent.state import AgentState
from agent.model import ChatModels
from agent.prompt.rewriter_prompt import get_rewriter_prompt
import traceback

logger = get_logger(__name__)

def rewriter_node(state: AgentState) -> dict:
    """
    Rewrites the user query to optimize for better vector search results.
    Increments the iteration_count.
    """
    query = state.get("query", "")
    iteration_count = state.get("iteration_count", 0)
    
    logger.info(f"Rewriting query for better retrieval (Iteration {iteration_count + 1}). Original: {query}")
    
    llm = ChatModels.get_nemotron3super()
    prompt = get_rewriter_prompt()
    chain = prompt | llm | StrOutputParser()
    
    try:
        new_query = chain.invoke({"query": query}).strip(' "\'')
        logger.info(f"Rewritten Query: {new_query}")
        log_node_event("rewriter_node", "SUCCESS")
        
        return {
            "query": new_query,
            "iteration_count": iteration_count + 1
        }
    except Exception as e:
        logger.error(f"Rewriter failed: {e}")
        log_system_error(traceback.format_exc())
        return {"iteration_count": iteration_count + 1}
