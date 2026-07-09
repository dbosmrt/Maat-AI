from agent.utils.logger import get_logger, log_node_event, log_system_error
from langchain_core.output_parsers import JsonOutputParser
from agent.state import AgentState, DocumentRanking
from agent.model import ChatModels
from agent.prompt.reranker_prompt import get_reranker_prompt
import traceback

logger = get_logger(__name__)

def reranker_node(state: AgentState) -> dict:
    """
    Takes the retrieved documents and uses the LLM to filter out irrelevant chunks.
    This drastically reduces hallucinations in the final generation step.
    """
    query = state.get("query", "")
    documents = state.get("documents", [])
    
    if not query or not documents:
        logger.warning("reranker_node: Missing query or documents.")
        return {"documents": documents}
        
    logger.info(f"Re-ranking {len(documents)} retrieved documents...")
    
    # Initialize LLM and Parser
    llm = ChatModels.get_nemotron3super()
    structured_llm = llm.with_structured_output(DocumentRanking)
    
    # Format the documents for the prompt
    docs_text = ""
    for i, doc in enumerate(documents):
        docs_text += f"\n--- Document Index: {i} ---\n{doc}\n"
        
    prompt = get_reranker_prompt()
    chain = prompt | structured_llm
    
    try:
        ranking = chain.invoke({
            "query": query, 
            "docs_text": docs_text,
            "format_instructions": "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
        })
        
        ranking_dict = ranking.dict() if hasattr(ranking, "dict") else dict(ranking)
        relevant_indices = ranking_dict.get("relevant_indices", [])
        
        # Filter the original documents list based on the returned indices
        filtered_docs = []
        for idx in relevant_indices:
            if 0 <= idx < len(documents):
                filtered_docs.append(documents[idx])
                
        logger.info(f"Re-ranker kept {len(filtered_docs)} out of {len(documents)} documents.")
        
        log_node_event("reranker_node", "SUCCESS")
        
        return {
            "documents": filtered_docs
        }
        
    except Exception as e:
        logger.error(f"Re-ranker node failed: {e}")
        log_system_error(traceback.format_exc())
        log_node_event("reranker_node", "PARSING_RETRY", error_payload=str(e))
        # Fallback: if the LLM fails, just return all documents so the pipeline doesn't break
        return {"documents": documents}
