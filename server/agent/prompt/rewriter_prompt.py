from langchain_core.prompts import ChatPromptTemplate

def get_rewriter_prompt(iteration_count: int = 0) -> ChatPromptTemplate:
    """
    Get a rewriter prompt tailored to the specific iteration/retry attempt.
    
    Args:
        iteration_count: Current iteration number (0 = first attempt, 1 = first retry, etc.)
    
    Returns:
        ChatPromptTemplate with strategy appropriate for this iteration
    """
    # Base instruction that applies to all iterations
    base_instruction = """You are an expert legal query re-writer. 
The user's original query did not retrieve relevant legal statutes from the database.
Your job is to rewrite the query to make it highly optimized for semantic vector search.
Focus on extracting core legal concepts, stripping conversational filler, and using synonyms if the original terms might not exactly match legal texts.

Return ONLY the rewritten query string. Do NOT add any conversational text or quotes around the output."""
    
    # Strategy-specific additions based on iteration
    if iteration_count == 0:
        # First attempt (initial query decomposition already happened) - focus on core extraction
        strategy = """
        
STRATEGY FOR THIS ATTEMPT: Initial optimization
- Extract the core legal intent from the user's query
- Remove conversational filler and emotional language  
- Use precise legal terminology where applicable
- Focus on the essential legal question being asked"""
    
    elif iteration_count == 1:
        # First retry - focus on terminology standardization
        strategy = """
        
STRATEGY FOR THIS ATTEMPT: Terminology standardization  
- Convert colloquial phrases to formal legal terms (e.g., "kicked out" → "illegal eviction", "took my stuff" → "theft of property")
- Use standard legal synonyms for key concepts
- Ensure proper naming of legal concepts and procedures
- Focus on making the query speak the language of legal texts"""
        
    elif iteration_count == 2:
        # Second retry - add implicit legal concepts
        strategy = """
        
STRATEGY FOR THIS ATTEMPT: Implicit concept enrichment
- Identify and add commonly associated legal elements that may not be explicitly stated
- For property crimes: consider adding "intent", "ownership", "possession" 
- For violent crimes: consider adding "harm", "threat", "fear"
- For contractual issues: consider adding "breach", "damages", "consideration"
- Think about what legal doctrines or elements are typically required to prove the core issue"""
        
    else:
        # Third+ retry - broaden scope and consider alternatives
        strategy = """
        
STRATEGY FOR THIS ATTEMPT: Scope broadening and alternative approaches
- Consider related statutes or offenses that might apply
- Think about alternative legal theories or interpretations
- Include broader categories or parent concepts
- Consider procedural aspects or jurisdictional elements
- Think about analogous situations that might be covered by similar laws"""
    
    full_prompt = base_instruction + strategy
    
    return ChatPromptTemplate.from_messages([
        ("system", full_prompt),
        ("user", "Original query: {query}")
    ])
