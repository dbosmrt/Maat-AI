from langchain_core.prompts import ChatPromptTemplate

RERANKER_SYSTEM_PROMPT = """You are an expert legal AI Re-ranker.
Your job is to read a list of retrieved legal documents and identify WHICH documents are actually relevant to answering the user's query.

CRITICAL RULES:
- Return a raw JSON object only. Do NOT include markdown blocks (like ```json), inline markdown bold formatting (like **), or trailing explanatory text.
- Read each document carefully.
- If a document contains laws, sections, or concepts that help answer the query, include its index.
- If a document is completely unrelated, ignore it.
- If NO documents are relevant, return an empty list.
- Do NOT hallucinate or guess document indices. Only include indices that explicitly exist in the provided list.

{format_instructions}
"""

def get_reranker_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", RERANKER_SYSTEM_PROMPT),
        ("user", "Query: {query}\n\nRetrieved Documents:\n{docs_text}")
    ])
