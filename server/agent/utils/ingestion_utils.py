from agent.utils.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

def validate_paths(input_dir: str, output_dir: str) -> bool:
    """Checks if directories exist and ensures output directory is created."""
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        logger.error("Input directory does not exist or is invalid: %s", input_dir)
        return False
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return True

def parse_pdf_with_docling(file_path: str, output_dir: str) -> bool:
    """Parses PDF with Docling and saves to Markdown."""
    try:
        from langchain_docling import DoclingLoader
        logger.info("Attempting to parse %s using DoclingLoader...", file_path)
        loader = DoclingLoader(file_path=file_path)
        docs = loader.load()
        md_content = "\n\n".join(doc.page_content for doc in docs)
        
        return _save_markdown(file_path, output_dir, md_content)
    except ImportError:
        logger.warning("langchain_docling is not installed.")
        return False
    except Exception as e:
        logger.warning("Docling failed: %s", e)
        return False

def parse_pdf_with_unstructured(file_path: str, output_dir: str) -> bool:
    """Parses PDF with Unstructured and saves to Markdown."""
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        logger.info("Attempting to parse %s using UnstructuredPDFLoader...", file_path)
        loader = UnstructuredPDFLoader(file_path=file_path, mode="elements")
        docs = loader.load()
        
        md_content = ""
        for doc in docs:
            category = doc.metadata.get("category", "")
            text = doc.page_content.strip()
            if not text:
                continue
            if category == "Title":
                md_content += f"\n## {text}\n\n"
            elif category == "ListItem":
                md_content += f"- {text}\n"
            else:
                md_content += f"{text}\n\n"
                
        return _save_markdown(file_path, output_dir, md_content)
    except ImportError:
        logger.warning("langchain-community or unstructured is not installed.")
        return False
    except Exception as e:
        logger.error("Unstructured parser failed: %s", e)
        return False

def _save_markdown(file_path: str, output_dir: str, md_content: str) -> bool:
    """Helper to save markdown content to file."""
    path_obj = Path(file_path)
    output_path = Path(output_dir)
    md_file_path = output_path / f"{path_obj.stem}.md"
    try:
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Successfully saved Markdown to %s", md_file_path)
        return True
    except IOError as e:
        logger.error("Failed to write Markdown file %s: %s", md_file_path, e)
        return False
