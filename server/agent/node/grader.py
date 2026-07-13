from agent.utils.logger import get_logger, log_node_event, log_system_error, LOG_DIR
from langchain_core.output_parsers import JsonOutputParser
from agent.state import AgentState, DocumentGrade
from agent.model import ChatModels
from agent.prompt.grader_prompt import get_grader_prompt
import os
import traceback

logger = get_logger(__name__)

# Circuit-breaker: max retry iterations before falling back to web search
MAX_RETRIEVAL_RETRIES = 2

# Minimum score to accept LLM's is_relevant=True verdict.
# Below this, we override to False regardless of LLM output.
MIN_RELEVANCE_SCORE_OVERRIDE = 0.5

def grader_node(state: AgentState) -> dict:
    """
    Evaluates if the retrieved documents are sufficient to answer the query.
    If not, or if the user explicitly requested case laws, it routes to web search.
    """
    query = state.get("query", "")
    documents = state.get("documents", [])
    requires_case_law = state.get("requires_case_law", False)
    
    if not query:
        logger.warning("grader_node: Missing query.")
        return {"search_required": False, "retry_retrieval": False}
        
    # Heuristic 1: If user wants Case Law, our DB (which only has statutes) won't have it.
    if requires_case_law:
        logger.info("Grader Node: Query requires external case law. Routing to Web Search.")
        return {"search_required": True, "retry_retrieval": False}
        
    # Heuristic 2: If no documents survived the re-ranker, we must search.
    if not documents:
        iteration_count = state.get("iteration_count", 0)
        if iteration_count < MAX_RETRIEVAL_RETRIES:
            logger.info(f"Grader Node: No documents provided (Iteration {iteration_count}). Routing to Rewriter.")
            return {"search_required": False, "retry_retrieval": True}
        else:
            logger.info("Grader Node: No documents provided. Max retries reached. Routing to Web Search.")
            return {"search_required": True, "retry_retrieval": False}
        
    logger.info("Grader Node: Evaluating document relevance...")
    
    llm = ChatModels.get_sarvam_m()
    structured_llm = llm.with_structured_output(DocumentGrade)
    
    # Format documents
    docs_text = "\n".join(documents)
    
    prompt = get_grader_prompt()
    chain = prompt | structured_llm
    
    try:
        grade = chain.invoke({
            "query": query,
            "docs_text": docs_text,
            "format_instructions": "Format: STRICT JSON MATCH. DO NOT USE MARKDOWN."
        })
        
        grade_dict = grade.dict() if hasattr(grade, "dict") else dict(grade)
        is_relevant = grade_dict.get("is_relevant", False)
        relevance_score = grade_dict.get("context_relevance_score", 1.0)
        diversity_analysis = grade_dict.get("chunk_diversity", "")
        
        logger.info(f"Grader Evaluation -> Relevance: {is_relevant} | Score: {relevance_score} | Diversity: {diversity_analysis}")
        
        # Code-level safety override: if the LLM said relevant but score is
        # below our minimum threshold, override to irrelevant. This prevents
        # the permissive grading prompt from making the retry loop a no-op.
        if is_relevant and relevance_score < MIN_RELEVANCE_SCORE_OVERRIDE:
            logger.warning(
                f"Grader Override: LLM said is_relevant=True but score={relevance_score} "
                f"< {MIN_RELEVANCE_SCORE_OVERRIDE}. Overriding to is_relevant=False."
            )
            is_relevant = False
        
        # Health Check Auditing
        if relevance_score < 0.4:
            warning_msg = f"[{query}] Critical Context Starvation! Score: {relevance_score}. Diversity: {diversity_analysis}"
            logger.warning(warning_msg)
            health_log = os.path.join(LOG_DIR, "retrieval_health_warnings.log")
            with open(health_log, "a") as f:
                f.write(warning_msg + "\n")
                
        if is_relevant:
            logger.info("Grader Node: Documents are highly relevant. Bypassing Web Search.")
            log_node_event("grader_node", "SUCCESS")
            return {"search_required": False, "retry_retrieval": False}
        else:
            iteration_count = state.get("iteration_count", 0)
            if iteration_count < MAX_RETRIEVAL_RETRIES:
                logger.info(f"Grader Node: Documents are irrelevant (Iteration {iteration_count}). Retrying retrieval loop.")
                log_node_event("grader_node", "RETRY")
                return {"search_required": False, "retry_retrieval": True}
            else:
                logger.info("Grader Node: Documents are irrelevant and max retries reached. Routing to Web Search.")
                log_node_event("grader_node", "SUCCESS")
                return {"search_required": True, "retry_retrieval": False}
            
    except Exception as e:
        logger.error(f"Grader node failed: {e}")
        log_system_error(traceback.format_exc())
        log_node_event("grader_node", "PARSING_RETRY", error_payload=str(e))
        # Safe fallback: try generating with what we have
        return {"search_required": False, "retry_retrieval": False}
