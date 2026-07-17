import os
from langchain_chroma import Chroma
# Pinecone import lazy-loaded to avoid hard dependency if not used
from agent.model import EmbeddingModels

# Define default vector store directory relative to this script
VECTOR_STORE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../vector_store")
)


class VectorDatabases:
    @staticmethod
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

    @staticmethod
    def get_pinecone_db(embeddings=None):
        """
        Initializes and returns the Pinecone vector store connection.
        Optionally accepts a pre-built embeddings instance (e.g. from fallback logic).
        """
        # Lazy import pinecone only when needed
        try:
            from langchain_pinecone import PineconeVectorStore
        except ImportError as e:
            raise ImportError(
                "Pinecone dependencies not installed. Install pinecone-client and langchain-pinecone to use Pinecone."
            ) from e

        if embeddings is None:
            embeddings = EmbeddingModels.get_nemotron_embed()

        api_key = os.environ.get("PINECONE_KEY")
        if not api_key:
            raise ValueError("PINECONE_KEY environment variable is not set")

        vectorstore = PineconeVectorStore(
            index_name="legal_rag",
            embedding=embeddings,
            pinecone_api_key=api_key,
        )
        return vectorstore


# Module-level convenience functions (for backward compatibility)
def get_vector_store(embeddings=None):
    return VectorDatabases.get_vector_store(embeddings)

def get_pinecone_db(embeddings=None):
    return VectorDatabases.get_pinecone_db(embeddings)
