from agent.utils.logger import get_logger, log_node_event, log_system_error
from agent.state import AgentState, RewriterOutput
from agent.model import ChatModels
from agent.prompt.rewriter_prompt import get_rewriter_prompt
import traceback
from typing import Union, cast

logger = get_logger(__name__)


def rewriter_node(state: AgentState) -> dict:
    """
    Rewrites the user query to optimize for better vector search results.
    Increments the iteration_count.
    """
    query = state.get("query", "")
    iteration_count = state.get("iteration_count", 0)

    logger.info(f"Rewriting query for better retrieval (Iteration {iteration_count + 1}). Original: {query}")

    llm = ChatModels.get_sarvam_m()
    # Pass iteration count to get the appropriate prompt strategy
    prompt = get_rewriter_prompt(iteration_count)
    structured_llm = llm.with_structured_output(RewriterOutput)
    chain = prompt | structured_llm

    try:
        # chain.invoke returns Union[dict, BaseModel] but we know it's dict or RewriterOutput
        result = chain.invoke({"query": query})
        result_typed = cast(Union[dict, RewriterOutput], result)
        # Type guard: result could be dict or RewriterOutput depending on LLM implementation
        if isinstance(result_typed, dict):
            new_query = result_typed.get("rewritten_query", query).strip(' "\'')
        else:
            new_query = result_typed.rewritten_query.strip(' "\'')
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
