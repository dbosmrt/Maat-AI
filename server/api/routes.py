import os
import json
import uuid
import time
from typing import List, Tuple, Any
from fastapi import APIRouter, HTTPException, Depends
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.messages.utils import get_buffer_string

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

from api.models import ChatRequest, ChatResponse, StartSessionResponse, ChatHistoryResponse, SessionItem, SessionListResponse
from api.security import get_api_key
from agent.chat_graph import build_chat_graph
from agent.state import AgentState
from agent.utils.logger import get_logger

logger = get_logger(__name__)

# Token limit for chat history (leave room for prompt + response)
_MAX_HISTORY_TOKENS = 3000
_ENCODING = "cl100k_base"  # GPT-4 / cl100k_base encoding


def _count_tokens(messages: List[BaseMessage]) -> int:
    """
    Count tokens in a list of messages using tiktoken if available,
    otherwise estimate from character count.
    """
    if _HAS_TIKTOKEN:
        try:
            encoding = tiktoken.get_encoding(_ENCODING)
            text = get_buffer_string(messages)
            return len(encoding.encode(text))
        except Exception:
            pass
    # Fallback: rough estimate ~4 chars per token
    text = get_buffer_string(messages)
    return max(1, len(text) // 4)


def _truncate_history_to_token_limit(
    messages: List[BaseMessage],
    max_tokens: int = _MAX_HISTORY_TOKENS
) -> List[BaseMessage]:
    """
    Truncate message history to fit within token limit.
    Keeps most recent messages first (sliding window from end).
    """
    if not messages:
        return []

    count = _count_tokens(messages)
    if count <= max_tokens:
        return messages

    # Sliding window from the end - keep most recent messages
    for i in range(1, len(messages) + 1):
        trimmed = messages[-i:]
        if _count_tokens(trimmed) <= max_tokens:
            return trimmed

    # If even 1 message exceeds, return just that
    logger.warning(
        "Single message exceeds token limit (%d tokens), returning as-is",
        _count_tokens([messages[-1]])
    )
    return [messages[-1]]

router = APIRouter()

# Lazy graph instance - compiled on first use
_chat_graph_holder = {"instance": None}


def get_chat_graph():
    """Get or create the compiled chat graph (lazy initialization)."""
    if _chat_graph_holder["instance"] is None:
        _chat_graph_holder["instance"] = build_chat_graph()
    return _chat_graph_holder["instance"]

CHAT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "chats"))
os.makedirs(CHAT_DIR, exist_ok=True)

def _load_session_file(session_id: str) -> dict | None:
    filepath = os.path.join(CHAT_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def _save_session_file(session_id: str, memory_summary: str, history: List[dict]):
    filepath = os.path.join(CHAT_DIR, f"{session_id}.json")
    data = {
        "session_id": session_id,
        "memory_summary": memory_summary,
        "history": history
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

@router.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "Ma-at Legal AI API is running"}

@router.post("/api/v1/chat/start", response_model=StartSessionResponse)
def start_chat(api_key: str = Depends(get_api_key)):
    """Initializes a new chat session."""
    session_id = str(uuid.uuid4())[:8]
    _save_session_file(session_id, "", [])
    logger.info(f"New chat session created: {session_id}")
    return StartSessionResponse(session_id=session_id, message="Session started successfully.")

@router.get("/api/v1/chat/sessions", response_model=SessionListResponse)
def list_sessions(api_key: str = Depends(get_api_key)):
    """Lists all available chat sessions with a preview of the first user message."""
    sessions = []
    for filename in os.listdir(CHAT_DIR):
        if not filename.endswith(".json"):
            continue
        session_id = filename.replace(".json", "")
        data = _load_session_file(session_id)
        if data is None:
            continue
        history = data.get("history", [])
        preview = "New conversation"
        for msg in history:
            if msg.get("type") == "human":
                preview = msg["content"][:80]
                break
        sessions.append(SessionItem(
            session_id=session_id,
            preview=preview,
            message_count=len(history)
        ))
    return SessionListResponse(sessions=sessions)

@router.delete("/api/v1/chat/{session_id}")
def delete_session(session_id: str, api_key: str = Depends(get_api_key)):
    """Deletes a chat session file."""
    filepath = os.path.join(CHAT_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        logger.warning(f"Attempted to delete non-existent session: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    os.remove(filepath)
    logger.info(f"Chat session deleted: {session_id}")
    return {"status": "ok", "message": f"Session {session_id} deleted."}

@router.get("/api/v1/chat/{session_id}", response_model=ChatHistoryResponse)
def get_chat_history(session_id: str, api_key: str = Depends(get_api_key)):
    """Retrieves the message history for a given session."""
    data = _load_session_file(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")

    return ChatHistoryResponse(
        session_id=session_id,
        history=data.get("history", [])
    )

@router.post("/api/v1/chat/{session_id}", response_model=ChatResponse)
def invoke_chat(session_id: str, request: ChatRequest, api_key: str = Depends(get_api_key)):
    """Core RAG inference endpoint. Use 'new' as the session_id to automatically generate a new chat session."""
    start_time = time.perf_counter()

    if session_id.lower() == "new":
        session_id = str(uuid.uuid4())[:8]
        # Initialize an empty session file
        _save_session_file(session_id, "", [])
        data: dict | None = {"session_id": session_id, "memory_summary": "", "history": []}
        logger.info(f"Auto-created new session: {session_id}")
    else:
        data = _load_session_file(session_id)
        if not data:
            logger.warning(f"Chat invoked on non-existent session: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found. Provide a valid session ID or use 'new'.")

    if data is None:
        raise HTTPException(status_code=500, detail="Failed to load session data")

    logger.info(f"Chat invoked on session {session_id}: '{request.query[:80]}...'")

    memory_summary = str(data.get("memory_summary", ""))
    raw_history_any = data.get("history", [])
    raw_history: List[dict] = raw_history_any if isinstance(raw_history_any, list) else []

    # Rebuild LangChain message objects
    lc_history: List[Any] = []
    for msg in raw_history:
        if msg["type"] == "human":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            lc_history.append(AIMessage(content=msg["content"]))

    # Append the new user query
    lc_history.append(HumanMessage(content=request.query))

    # Token-aware history truncation
    # Use tiktoken to count tokens and keep history within budget
    max_history_tokens = int(os.environ.get("MAX_HISTORY_TOKENS", "4000"))

    def _count_message_tokens(messages: List[Any]) -> int:
        """Count tokens in a list of messages using tiktoken."""
        try:
            import tiktoken as tiktoken_module

            # Use cl100k_base which is compatible with most modern models
            encoding = tiktoken_module.get_encoding("cl100k_base")
            total = 0
            for msg in messages:
                # Count tokens in message content
                total += len(encoding.encode(msg.content))
                # Add overhead for message role (roughly 4 tokens per message)
                total += 4
            return total
        except ImportError:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return sum(len(getattr(m, "content", "")) for m in messages) // 4

    def _truncate_history_by_tokens(messages: List[Any], max_tokens: int) -> List[Any]:
        """Truncate messages from the start to fit within token budget."""
        if not messages:
            return []

        # Count total tokens
        total_tokens = _count_message_tokens(messages)

        if total_tokens <= max_tokens:
            return messages

        # Remove oldest messages until we fit
        while messages and total_tokens > max_tokens:
            messages.pop(0)
            # Recalculate (could be optimized by subtracting, but this is safer)
            total_tokens = _count_message_tokens(messages)

        logger.debug(
            f"Truncated history from {len(messages) + (total_tokens - max_tokens) // 4} "
            f"to {len(messages)} messages ({total_tokens} tokens, budget {max_tokens})"
        )
        return messages

    # Get token-limited history
    lc_history = _truncate_history_by_tokens(lc_history, max_history_tokens)

    # Build the initial state for the graph
    state: AgentState = {
        "session_id": session_id,
        "chat_history": lc_history,
        "memory_summary": memory_summary,
        "query": request.query,
        "decomposed_query": {},
        "law_domain": "General",
        "is_scenario": False,
        "is_general_chat": False,
        "requires_case_law": False,
        "search_required": False,
        "retry_retrieval": False,
        "documents": [],
        "case_laws": [],
        "generation": "",
        "iteration_count": 0,
        "ingest_input_dir": "",
        "ingest_output_dir": "",
        "ingest_status": "",
    }

    # Invoke Graph
    try:
        result = get_chat_graph().invoke(state)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Pipeline failed after {elapsed_ms:.0f}ms for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    generation = result.get("generation", "Failed to generate response.")
    domain = result.get("law_domain", "General")

    # Save the new history
    raw_history.append({"type": "human", "content": request.query})
    raw_history.append({"type": "ai", "content": generation})

    # In a real system, the memory summary would be updated here.
    _save_session_file(session_id, memory_summary, raw_history)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Chat completed for session {session_id} in {elapsed_ms:.0f}ms | domain={domain}")

    return ChatResponse(
        session_id=session_id,
        generation=generation,
        law_domain=domain
    )
