from agent.utils.logger import get_logger
import time
from agent.state import AgentState
from agent.model import EmbeddingModels
from agent.utils.embedding_utils import get_vector_store
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = get_logger(__name__)

def reciprocal_rank_fusion(results_lists, k=60):
    fused_scores = {}
    for docs in results_lists:
        for rank, doc in enumerate(docs):
            # Use page_content as the unique identifier for simplicity
            if doc.page_content not in fused_scores:
                fused_scores[doc.page_content] = {"doc": doc, "score": 0.0}
            fused_scores[doc.page_content]["score"] += 1.0 / (rank + k)
            
    reranked_results = [
        item["doc"] for item in sorted(
            fused_scores.values(), key=lambda x: x["score"], reverse=True
        )
    ]
    return reranked_results

_cached_bm25_retriever = None

def invalidate_bm25_cache():
    """Clears the cached BM25 retriever so it is rebuilt on next query.
    Call this after ingesting new documents into ChromaDB."""
    global _cached_bm25_retriever
    _cached_bm25_retriever = None
    logger.info("BM25 cache invalidated. Will rebuild on next retrieval.")

def _get_bm25_retriever(vectorstore):
    global _cached_bm25_retriever
    if _cached_bm25_retriever is None:
        logger.info("Initializing BM25 keyword index from ChromaDB... (this happens once)")
        # Get all documents to build the BM25 index
        try:
            docs_data = vectorstore.get(include=["documents", "metadatas"])
            lc_docs = []
            for content, meta in zip(docs_data["documents"], docs_data["metadatas"]):
                context_path = meta.get("context_path", "")
                enriched_content = f"{context_path}\n{content}"
                lc_docs.append(Document(page_content=enriched_content, metadata=meta))
                
            _cached_bm25_retriever = BM25Retriever.from_documents(lc_docs)
            _cached_bm25_retriever.k = 20
            logger.info("BM25 index initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise e
    return _cached_bm25_retriever

def retriever_node(state: AgentState) -> dict:
    """
    Retrieves the top k most relevant document chunks from ChromaDB based on the user's query.
    Uses semantic search with automatic embedding model fallback and exponential backoff.
    On retry iterations (iteration_count > 0), uses the rewritten raw query directly
    since the decomposed_query is stale from the first pass.
    """
    query = state.get("query", "")
    decomposed = state.get("decomposed_query", {})
    iteration_count = state.get("iteration_count", 0)
    
    if not query:
        logger.warning("retriever_node: No query found in state.")
        return {"documents": []}
    
    # On retry iterations, the rewriter has rewritten the raw query but
    # decomposed_query is stale from pass 1. Use raw query directly.
    if iteration_count > 0:
        hybrid_query = query
        logger.info(f"Retry iteration {iteration_count}: using rewritten raw query for retrieval.")
    else:
        semantic = decomposed.get("semantic_focus", "")
        statutory = decomposed.get("statutory_focus", "")
        procedural = decomposed.get("procedural_focus", "")
        # Construct a hybrid search string optimized for both sparse and dense layers
        hybrid_query = f"{semantic} {statutory} {procedural}".strip()
        if not hybrid_query:
            hybrid_query = query
        
    logger.info(f"Retrieving context for hybrid query: '{hybrid_query}'")
    
    # --- Attempt 1: Use primary embedding model with exponential backoff ---
    max_retries = 5
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            vectorstore = get_vector_store()  # Uses primary model
            
            # Setup Dense Retriever
            dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
            
            # Setup Sparse Retriever
            bm25_retriever = _get_bm25_retriever(vectorstore)
            
            # Execute parallel queries
            dense_docs = dense_retriever.invoke(hybrid_query)
            sparse_docs = bm25_retriever.invoke(hybrid_query)
            
            # Fuse them via RRF
            docs = reciprocal_rank_fusion([dense_docs, sparse_docs])
            docs = docs[:20] # Guarantee strict limit
            
            logger.info(f"Successfully retrieved {len(docs)} chunks (Native RRF, primary, attempt {attempt+1}).")
            return _format_docs(docs)
        except Exception as e:
            delay = base_delay * (2 ** attempt)  # 2s, 4s, 8s, 16s, 32s
            if attempt < max_retries - 1:
                logger.warning(f"Primary embed failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Primary embedding model exhausted all {max_retries} retries.")
    
    # --- Attempt 2: Use fallback embedding model ---
    logger.info("Switching to fallback embedding model...")
    try:
        fallback_embeddings = EmbeddingModels.get_embed_with_fallback()
        vectorstore = get_vector_store(embeddings=fallback_embeddings)
        
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        bm25_retriever = _get_bm25_retriever(vectorstore)
        
        dense_docs = dense_retriever.invoke(hybrid_query)
        sparse_docs = bm25_retriever.invoke(hybrid_query)
        docs = reciprocal_rank_fusion([dense_docs, sparse_docs])
        docs = docs[:20]
        
        logger.info(f"Successfully retrieved {len(docs)} chunks (Native RRF, fallback model).")
        return _format_docs(docs)
    except Exception as e:
        logger.error(f"Fallback embedding model also failed: {e}")
        return {"documents": []}


def _format_docs(docs) -> dict:
    """
    Formats retrieved LangChain Documents into strings with metadata prefixes.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        context_path = doc.metadata.get("context_path", "Unknown Source")
        formatted_text = f"[Source: {context_path}]\n{doc.page_content}"
        formatted_docs.append(formatted_text)
        logger.debug(f"Retrieved chunk {i+1} from {context_path}")
        
    return {"documents": formatted_docs}
