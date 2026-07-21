import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../server')))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

# Import the Pydantic models used by the nodes
from agent.state import (
    QueryClassification,
    DocumentRanking,
    DocumentGrade,
    SearchQueries,
    DecomposedQuery,
)


# --- Factory functions that create proper Pydantic model instances ---

def make_query_classification(prompt_text: str = "") -> QueryClassification:
    law_domain = "General"
    is_scenario = False
    requires_case_law = False
    is_general_chat = False

    # Extract only the user query from the prompt (last HumanMessage)
    user_query = ""
    if "HumanMessage(content='" in prompt_text:
        # Parse the user message content
        parts = prompt_text.split("HumanMessage(content='")
        if len(parts) > 1:
            user_query = parts[-1].split("'")[0]
    elif "{query}" in prompt_text:
        # Fallback - extract from format
        pass

    search_text = user_query if user_query else prompt_text

    if "Bharatiya Nyaya Sanhita" in search_text or "anticipatory bail" in search_text:
        law_domain = "Criminal"
    elif "contract under coercion" in search_text or "neighbor built a fence" in search_text:
        law_domain = "Civil"

    if "contract under coercion" in search_text or "neighbor built a fence" in search_text:
        is_scenario = True

    if "Supreme Court judgments" in search_text or "case studies" in search_text:
        requires_case_law = True

    if "Chief Justice of India" in search_text or "hello" in search_text.lower() or "how are you" in search_text.lower():
        is_general_chat = True

    return QueryClassification(
        law_domain=law_domain,
        is_scenario=is_scenario,
        requires_case_law=requires_case_law,
        is_general_chat=is_general_chat,
    )


def make_document_ranking(prompt_text: str = "") -> DocumentRanking:
    relevant_indices = []
    if "The punishment for theft is imprisonment" in prompt_text:
        relevant_indices.append(1)
    if "theft in a dwelling house" in prompt_text:
        relevant_indices.append(3)
    return DocumentRanking(relevant_indices=relevant_indices)


def make_document_grade(prompt_text: str = "") -> DocumentGrade:
    is_relevant = True
    if "A person who commits murder" in prompt_text:
        is_relevant = False

    return DocumentGrade(
        is_relevant=is_relevant,
        chunk_diversity="Mocked diversity analysis",
        context_relevance_score=0.8 if is_relevant else 0.2,
        failure_reason=None if is_relevant else "MISSING_KEY_CONCEPT",
    )


def make_search_queries() -> SearchQueries:
    return SearchQueries(search_queries=["mock query 1", "mock query 2"])


def make_decomposed_query() -> DecomposedQuery:
    return DecomposedQuery(
        semantic_focus="mock semantic",
        statutory_focus="mock statutory",
        procedural_focus="mock procedural",
        domain="criminal",
    )


def make_llm_invoke_response(prompt_text: str = "") -> AIMessage:
    if "punishment for theft" in prompt_text.lower() or "section 303" in prompt_text.lower():
        return AIMessage(
            content=(
                "According to [Source: BNS - CHAPTER XVII - Section 303], whoever commits theft "
                "shall be punished with imprisonment of either description for a term which may "
                "extend to three years, or with fine, or with both."
            )
        )
    return AIMessage(content="Mocked LLM response for the query.")


def make_runnable_llm_direct():
    """Create a Runnable that acts as the LLM for direct invocation (StrOutputParser chains)."""
    def invoke_fn(input_dict):
        # Handle both prompt dict and prompt string
        if isinstance(input_dict, dict):
            # Chain passes the formatted prompt dict
            prompt_text = str(input_dict)
        else:
            prompt_text = str(input_dict)
        return make_llm_invoke_response(prompt_text)
    return RunnableLambda(invoke_fn)


def make_runnable_structured(output_factory):
    """Create a Runnable for structured output (with_structured_output chains)."""
    def invoke_fn(input_dict):
        prompt_text = str(input_dict)
        return output_factory(prompt_text)
    return RunnableLambda(invoke_fn)


# --- Fixtures that patch at the module level where nodes import ChatModels ---

@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """Mock embeddings for all tests."""
    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [[0.1] * 1024]
    mock_emb.embed_query.return_value = [0.1] * 1024

    monkeypatch.setattr('agent.model.EmbeddingModels.get_nemotron_embed', lambda: mock_emb)
    monkeypatch.setattr('agent.model.EmbeddingModels.get_embed_with_fallback', lambda: mock_emb)


@pytest.fixture(autouse=True)
def mock_vector_store(monkeypatch):
    """Mock vector store for all tests."""
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Mock Document Content", metadata={"context_path": "mock.pdf"})
    ]
    mock_vs.as_retriever.return_value = mock_retriever
    monkeypatch.setattr('agent.utils.embedding_utils.VectorDatabases.get_vector_store', lambda *a, **k: mock_vs)
    monkeypatch.setattr('agent.node.retriever._get_bm25_retriever', lambda *a, **k: mock_retriever)


@pytest.fixture(autouse=True)
def mock_ddgs(monkeypatch):
    """Mock DDGS for web search."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [{
        "title": "Arnesh Kumar vs State of Bihar - Supreme Court Judgment",
        "href": "https://example.com/arnesh-kumar",
        "body": "In Arnesh Kumar vs State of Bihar, the Supreme Court laid down guidelines for arrest under Section 41A CrPC."
    }]
    # Make it a proper context manager
    mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs.__exit__ = MagicMock(return_value=False)

    # Patch at the module where it's used
    monkeypatch.setattr('agent.node.web_search.DDGS', lambda: mock_ddgs)
    # Also patch at source for completeness
    monkeypatch.setattr('duckduckgo_search.DDGS', lambda: mock_ddgs)


# --- Per-node test fixtures (autouse when test file matches) ---

@pytest.fixture(autouse=True)
def _mock_for_qualifier(request, monkeypatch):
    if 'test_qualifier' not in request.module.__name__:
        return

    mock_llm = make_runnable_llm_direct()
    # For structured output, create a proper Runnable
    def with_structured_output(schema):
        return make_runnable_structured(lambda p: make_query_classification(p))

    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output)

    monkeypatch.setattr('agent.node.qualifier.ChatModels.get_sarvam_m', lambda: mock_llm)


@pytest.fixture(autouse=True)
def _mock_for_reranker(request, monkeypatch):
    if 'test_reranker' not in request.module.__name__:
        return

    mock_llm = make_runnable_llm_direct()
    def with_structured_output(schema):
        return make_runnable_structured(lambda p: make_document_ranking(p))
    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output)

    monkeypatch.setattr('agent.node.reranker.ChatModels.get_sarvam_m', lambda: mock_llm)


@pytest.fixture(autouse=True)
def _mock_for_grader(request, monkeypatch):
    if 'test_grader' not in request.module.__name__:
        return

    mock_llm = make_runnable_llm_direct()
    def with_structured_output(schema):
        return make_runnable_structured(lambda p: make_document_grade(p))
    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output)

    monkeypatch.setattr('agent.node.grader.ChatModels.get_sarvam_m', lambda: mock_llm)


@pytest.fixture(autouse=True)
def _mock_for_generator(request, monkeypatch):
    if 'test_generator' not in request.module.__name__:
        return

    # Generator uses direct invoke with StrOutputParser chain
    mock_llm = make_runnable_llm_direct()
    monkeypatch.setattr('agent.node.generator.ChatModels.get_sarvam_m', lambda: mock_llm)


@pytest.fixture(autouse=True)
def _mock_for_web_search(request, monkeypatch):
    if 'test_web_search' not in request.module.__name__:
        return

    mock_llm = make_runnable_llm_direct()
    def with_structured_output(schema):
        def get_structured(prompt_text):
            if "distill it into 2-3 short, focused web search queries" in prompt_text:
                return make_search_queries()
            if "semantic_focus" in prompt_text or "statutory_focus" in prompt_text:
                return make_decomposed_query()
            return make_query_classification(prompt_text)
        return make_runnable_structured(get_structured)
    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output)

    # Patch at the model module level since web_search imports from there
    monkeypatch.setattr('agent.model.ChatModels.get_sarvam_m', lambda: mock_llm)


@pytest.fixture(autouse=True)
def _mock_for_query_decomposer(request, monkeypatch):
    if 'test_query_decomposer' not in request.module.__name__:
        return

    mock_llm = make_runnable_llm_direct()
    def with_structured_output(schema):
        return make_runnable_structured(lambda _: make_decomposed_query())
    mock_llm.with_structured_output = MagicMock(side_effect=with_structured_output)

    monkeypatch.setattr('agent.node.query_decomposer.ChatModels.get_sarvam_m', lambda: mock_llm)


# --- Mock torch for ingestion tests ---

@pytest.fixture
def mock_torch(monkeypatch):
    """Mock torch for ingestion tests that test GPU availability."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    monkeypatch.setattr('agent.node.ingestion.torch', mock_torch)
    monkeypatch.setattr('agent.utils.ingestion_utils.torch', mock_torch)
    return mock_torch


@pytest.fixture
def mock_torch_unavailable(monkeypatch):
    """Mock torch as unavailable for ingestion tests testing CPU fallback."""
    monkeypatch.setattr('agent.node.ingestion.torch', None, raising=False)
    monkeypatch.setattr('agent.utils.ingestion_utils.torch', None, raising=False)