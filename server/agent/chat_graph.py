from agent.utils.logger import get_logger
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.node.query_decomposer import query_decomposer_node
from agent.node.qualifier import qualifier_node
from agent.node.retriever import retriever_node
from agent.node.reranker import reranker_node
from agent.node.grader import grader_node
from agent.node.web_search import web_search_node
from agent.node.generator import generator_node
from agent.node.rewriter import rewriter_node

logger = get_logger(__name__)

def build_chat_graph():
    """
    Builds and compiles the advanced Self-Reflective RAG LangGraph for chat.
    Workflow: Qualifier -> Retriever -> Reranker -> Grader -> [Web Search (Optional)] -> Generator
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add all nodes to the graph
    workflow.add_node("query_decomposer", query_decomposer_node)
    workflow.add_node("qualifier", qualifier_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("rewriter", rewriter_node)
    
    # Define conditional edge logic for the grader
    def grader_conditional(state: AgentState) -> str:
        if state.get("retry_retrieval", False):
            return "rewriter"
        if state.get("search_required", False):
            return "web_search"
        return "generator"
        
    def qualifier_conditional(state: AgentState) -> str:
        if state.get("is_general_chat", False):
            logger.info("General Chat detected! Bypassing retrieval and routing directly to generator.")
            return "generator"
        return "retriever"
    
    # Build the execution sequence
    workflow.add_edge(START, "query_decomposer")
    workflow.add_edge("query_decomposer", "qualifier")
    
    # Conditional edge from qualifier
    workflow.add_conditional_edges(
        "qualifier",
        qualifier_conditional,
        {
            "generator": "generator",
            "retriever": "retriever"
        }
    )
    
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "grader")
    
    # Conditional edge from grader
    workflow.add_conditional_edges(
        "grader",
        grader_conditional,
        {
            "rewriter": "rewriter",
            "web_search": "web_search",
            "generator": "generator"
        }
    )
    
    # Retry loop
    workflow.add_edge("rewriter", "retriever")
    
    workflow.add_edge("web_search", "generator")
    workflow.add_edge("generator", END)
    
    # Compile the graph
    logger.info("Compiling Chat Graph...")
    return workflow.compile()
