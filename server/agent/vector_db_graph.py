"""Vector DB pipeline construction for the Legal RAG Chatbot.

Builds a StateGraph that converts documents into a Pinecone vector store.

Workflow:
    ingestion -> cleaning -> chunking -> embedding -> END
"""

from langgraph.graph import END, START, StateGraph

from .node.chunking import chunking_node
from .node.cleaning import cleaning_node
from .node.embedding import embedding_node
from .node.ingestion import ingestion_node
from .state import AgentState
from .utils.logger import get_logger

logger = get_logger(__name__)


def build_db_pipeline():
    """
    Construct and compile the ingestion StateGraph.

    Returns:
        A compiled `CompiledStateGraph` ready to be invoked with an AgentState.
    """
    workflow = StateGraph(AgentState)

    # Add each node into the graph
    workflow.add_node("ingestion_node", ingestion_node)
    workflow.add_node("cleaning_node", cleaning_node)
    workflow.add_node("chunking_node", chunking_node)
    workflow.add_node("embedding_node", embedding_node)

    # Build the graph by connecting the nodes in order.
    workflow.add_edge(START, "ingestion_node")
    workflow.add_edge("ingestion_node", "cleaning_node")
    workflow.add_edge("cleaning_node", "chunking_node")
    workflow.add_edge("chunking_node", "embedding_node")
    workflow.add_edge("embedding_node", END)

    logger.info("Compiling the Vector DB pipeline Graph")
    return workflow.compile()
