# `grader.py`

## Module Overview
Implements an automated LLM-as-a-Judge to evaluate the health and relevance of the retrieved context *before* the generator is allowed to see it.

## Role in Pipeline
Executes right after the `reranker`. It controls a conditional routing switch: if it fails the documents, the pipeline aborts generation and routes to Web Search.

## Function Breakdown

### `grader_node(state: AgentState) -> dict`
- **Inputs**: The filtered `documents` from the reranker.
- **Outputs**: `search_required` boolean flag in the state.
- **Logic**:
  - Uses `with_structured_output(DocumentGrade)`.
  - The LLM evaluates the context and outputs a `context_relevance_score` float (0.0 to 1.0).
  - If `is_relevant` is True (score >= 0.5), the state `search_required` is set to False, and the graph moves to the Generator.
  - If `is_relevant` is False, `search_required` is set to True, and the graph routes to the Web Search node.
  - **Permissive Tuning**: The grader prompt is heavily tuned to be *permissive* (0.4 threshold). Legal texts are dense, and overly strict graders cause infinite fallback loops. If a document mentions core legal concepts related to the query, it passes.
  - **Health Logging**: If the score is low, it triggers a system warning written to `retrieval_health_warnings.log`.
