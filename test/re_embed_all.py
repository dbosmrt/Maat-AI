"""
Re-embed ALL markdown files into ChromaDB.
Deletes the existing vector store and rebuilds from scratch.
"""
import os
import sys
import shutil
import logging
from agent.utils.logger import get_logger
from pathlib import Path

from agent.node.chunking import chunking_node
from agent.node.embedding import embedding_node, get_vector_store, VECTOR_STORE_DIR
from agent.state import AgentState


logger = get_logger(__name__)

MD_DIR = "data/markdown"

def main():
    md_path = Path(MD_DIR)
    md_files = sorted(md_path.glob("*.md"))
    
    if not md_files:
        logger.error(f"No markdown files found in {MD_DIR}")
        sys.exit(1)
        
    logger.info(f"Found {len(md_files)} markdown files: {[f.name for f in md_files]}")
    
    # Step 1: Wipe existing vector store
    if os.path.exists(VECTOR_STORE_DIR):
        logger.info(f"Deleting existing vector store at {VECTOR_STORE_DIR}...")
        shutil.rmtree(VECTOR_STORE_DIR)
    
    total_chunks = 0
    
    # Step 2: Process each file individually
    for md_file in md_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {md_file.name}")
        logger.info(f"{'='*60}")
        
        # Create a temp directory with just this one file for chunking
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copy(md_file, Path(temp_dir) / md_file.name)
            
            # Chunk
            state = AgentState(
                ingest_output_dir=temp_dir,
                documents=[],
                session_id="re-embed",
                chat_history=[],
                memory_summary="",
                query="",
                case_laws=[],
                generation="",
                iteration_count=0
            )
            
            chunk_result = chunking_node(state)
            chunks = chunk_result.get("documents", [])
            logger.info(f"  Chunks generated: {len(chunks)}")
            total_chunks += len(chunks)
            
            if not chunks:
                logger.warning(f"  Skipping {md_file.name} - no chunks generated.")
                continue
            
            # Embed
            state["documents"] = chunks
            embed_result = embedding_node(state)
            logger.info(f"  Embedding status: {embed_result.get('ingest_status')}")
    
    # Step 3: Verify
    logger.info(f"\n{'='*60}")
    logger.info("VERIFICATION")
    logger.info(f"{'='*60}")
    logger.info(f"Total chunks processed: {total_chunks}")
    
    vs = get_vector_store()
    db_count = vs._collection.count()
    logger.info(f"Total documents in ChromaDB: {db_count}")
    
    if db_count >= total_chunks * 0.9:
        logger.info("Re-embedding SUCCESSFUL. Vector store is populated.")
    else:
        logger.warning(f" Mismatch: expected ~{total_chunks}, got {db_count}. Some batches may have failed.")

if __name__ == "__main__":
    main()
