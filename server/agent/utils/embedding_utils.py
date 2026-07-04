import os
from langchain_chroma import Chroma
from agent.model import EmbeddingModels

# Define default vector store directory relative to this script
VECTOR_STORE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../vector_store")
)

def get_vector_store(embeddings=None) -> Chroma:
    """
    Initializes and returns the Chroma vector store connection.
    Optionally accepts a pre-built embeddings instance (e.g. from fallback logic).
    """
    if embeddings is None:
        embeddings = EmbeddingModels.get_nemotron_embed()
    
    # Ensure vector store directory exists
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    vectorstore = Chroma(
        collection_name="legal_rag",
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
    return vectorstore
