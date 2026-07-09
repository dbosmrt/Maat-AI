"""
Markdown Cleaning Node for Legal RAG Chatbot.

This module provides a LangGraph node that cleans the raw Markdown 
produced by the ingestion step.
"""

from agent.utils.logger import get_logger
from pathlib import Path

from agent.state import AgentState
from agent.utils.cleaning_utils import clean_markdown_text

logger = get_logger(__name__)


def cleaning_node(state: AgentState) -> dict:
    """
    LangGraph node to clean Markdown files before chunking.
    Reads .md files from ingest_output_dir, cleans them, and overwrites them.
    """
    md_dir = state.get("ingest_output_dir", "")
    
    if not md_dir:
        logger.error("cleaning_node failed: ingest_output_dir not found.")
        return {"ingest_status": "Failed: Missing markdown directory in state"}
        
    md_path = Path(md_dir)
    if not md_path.exists() or not md_path.is_dir():
        logger.error("Markdown directory not found: %s", md_dir)
        return {"ingest_status": "Failed: Invalid markdown directory"}
        
    md_files = list(md_path.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s", md_dir)
        return {"ingest_status": "Cleaning Completed (No files)"}
        
    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            cleaned_content = clean_markdown_text(content)
            
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
                
            logger.info("Cleaned and formatted: %s", md_file.name)
        except Exception as e:
            logger.error("Failed to clean file %s: %s", md_file.name, e)
            
    return {"ingest_status": "Cleaning Completed Successfully"}
