"""
LangGraph State definition for the Legal RAG Chatbot.
"""

import operator
from typing import TypedDict, List, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class DecomposedQuery(BaseModel):
    semantic_focus: str = Field(description="Abstract legal principles or narrative keywords (e.g., 'group robbery armed break-in at night').")
    statutory_focus: str = Field(description="Expected legislative shorthand or numbers (e.g., 'BNS 310 312 section').")
    procedural_focus: str = Field(description="Operational terms (e.g., 'electronic registration FIR evidence certification').")
    domain: Literal["criminal", "civil", "family"] = Field(description="Inferred legal domain.")

class QueryClassification(BaseModel):
    law_domain: Literal["Criminal", "Civil", "General"] = Field(
        description="Classify as 'Criminal' (e.g., theft, murder, bail, criminal codes like BNS/BNSS/BSA), 'Civil' (e.g., contracts, property disputes, divorce), or 'General' (e.g., general knowledge, names of judges)."
    )
    is_scenario: bool = Field(
        description="True if the user is describing a specific situation, event, or incident involving people (e.g. 'A slapped B', 'My neighbor did X', 'I signed a contract'). False if it is a general legal question (e.g. 'what is theft', 'what is the punishment for murder')."
    )
    requires_case_law: bool = Field(
        description="STRICTLY True ONLY if the user explicitly asks for judicial cases, supreme court judgments, past rulings, or case studies. STRICTLY False if they just ask for laws, punishments, or general advice."
    )
    is_general_chat: bool = Field(
        description="True if the user is just saying a greeting (e.g., 'hi', 'how are you') or asking a broad non-legal question. False if they are asking a specific legal or factual question."
    )

class DocumentRanking(BaseModel):
    relevant_indices: List[int] = Field(
        description="A list of integers representing the indices (0 to N-1) of the documents that are highly relevant to answering the query. If a document is completely irrelevant, do NOT include its index. If NO documents are relevant, return an empty list []."
    )

class DocumentGrade(BaseModel):
    is_relevant: bool = Field(
        description="True if the provided documents completely contain the answer to the user's query. False if the documents are irrelevant or missing critical information to answer the question."
    )
    chunk_diversity: str = Field(
        description="A short analysis evaluating if the retrieved chunks contain a healthy balance of diverse source sections."
    )
    context_relevance_score: float = Field(
        description="A score from 0.0 to 1.0 based on how well the chunks align with the user query."
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description="If the documents are not relevant, the reason why (e.g., MISSING_KEY_CONCEPT, INSUFFICIENT_CONTEXT_DEPTH, WRONG_JURISDICTION_FOCUS, TEMPORAL_MISMATCH, AMBIGUOUS_QUERY)."
    )

class SearchQueries(BaseModel):
    search_queries: List[str] = Field(
        description="A list of 2-3 short, focused search queries (each under 15 words) extracted from the user's legal question. Each query should target a different aspect of the question. Use legal terminology."
    )


class AgentState(TypedDict):
    """
    The state schema for the Legal RAG Chatbot LangGraph.
    """
    # Chat & Memory
    session_id: str
    chat_history: Annotated[List[BaseMessage], operator.add]
    memory_summary: str
    
    # Query Processing
    query: str
    decomposed_query: dict  # Output of query_decomposer
    law_domain: str  # e.g., 'Criminal', 'Civil', 'General'
    is_scenario: bool
    is_general_chat: bool
    requires_case_law: bool
    search_required: bool
    retry_retrieval: bool
    
    # RAG Context
    documents: List[str]  # Extracted text/chunks from Vector DB
    case_laws: List[str]  # Retrieved case laws
    
    # Orchestration & Output
    generation: str
    iteration_count: int
    
    # Ingestion specific (if used as an ingestion graph)
    ingest_input_dir: str
    ingest_output_dir: str
    ingest_status: str
