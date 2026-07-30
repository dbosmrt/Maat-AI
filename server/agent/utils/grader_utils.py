"""Utility functions for the Grader node in the Legal RAG Chatbot."""

from typing import Dict


def calculate_max_retries(decomposed_query: Dict) -> int:
    """
    Calculate the dynamic maximum number of retrieval retries based on query complexity.

    Args:
        decomposed_query: Dictionary containing the decomposed query components:
            - semantic_focus: Abstract legal principles or narrative keywords
            - statutory_focus: Expected legislative shorthand or numbers
            - procedural_focus: Operational terms
            - domain: Inferred legal domain ("criminal", "civil", or "family")

    Returns:
        Integer representing the maximum number of retries allowed (1-4).
        Base is 1 retry, with bonuses for complexity factors.
    """
    # Start with base retry count
    max_retries = 1

    # Bonus for lengthy semantic description (complex scenario)
    semantic = decomposed_query.get("semantic_focus", "")
    if len(semantic) > 50:  # More than 50 chars indicates complex narrative
        max_retries += 1

    # Bonus for missing statutory anchor (harder to ground the search)
    statutory = decomposed_query.get("statutory_focus", "")
    if not statutory.strip():  # No statutory reference provided
        max_retries += 1

    # Bonus for procedural complexity (multi-step legal process)
    procedural = decomposed_query.get("procedural_focus", "")
    if procedural.strip():  # Procedural terms present
        max_retries += 1

    # Cap the retries to prevent infinite loops (max 4 attempts total)
    return min(max_retries, 4)