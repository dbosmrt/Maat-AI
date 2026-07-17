from agent.utils.logger import get_logger
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState 

from agent.node.ingestion import ingestion_node
from agent.node.cleaning import cleaning_node
from agent.node.chunking import chunking_node
from agent.node.embedding import embedding_node

logger = get_logger(__name__)



def build_db_pipeline():
    """
    Builds a Pipeline to convert given document into vector store.
    Workflow : ingestion -> cleaning -> chunking -> embedding -> END.
    """

    workflow = StateGraph(AgentState)

    # Add each node into the graph 
    workflow.add_node("ingestion_node", ingestion_node)
    workflow.add_node("cleaning_node", cleaning_node)
    workflow.add_node("chunking_node", chunking_node)
    workflow.add_node("embedding_node", embedding_node)

    # now build the graph by connecting the nodes.
    workflow.add_edge(START, "ingestion_node")
    workflow.add_edge("ingestion_node", "cleaning_node")
    workflow.add_edge("cleaning_node", "chunking_node")
    workflow.add_edge("chunking_node", "embedding_node")
    workflow.add_edge("embedding_node", END)

    logger.info("Compiling the Vector DB pipeline Graph")
    return workflow.compile()

