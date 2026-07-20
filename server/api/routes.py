import os
import json
import uuid
import time
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from langchain_core.messages import HumanMessage, AIMessage

from api.models import ChatRequest, ChatResponse, StartSessionResponse, ChatHistoryResponse, SessionItem, SessionListResponse
from api.security import get_api_key
from agent.chat_graph import build_chat_graph
from agent.state import AgentState
from agent.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global graph instance for the API
chat_graph = build_chat_graph()

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
    from typing import Any
    lc_history: List[Any] = []
    for msg in raw_history:
        if msg["type"] == "human":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            lc_history.append(AIMessage(content=msg["content"]))

    # Append the new user query
    lc_history.append(HumanMessage(content=request.query))

    # Construct AgentState
    state = AgentState(
        query=request.query,
        session_id=session_id,
        chat_history=lc_history[-4:], # limit context
        memory_summary=memory_summary,
        is_scenario=False,
        requires_case_law=False,
        search_required=False,
        documents=[],
        case_laws=[],
        generation="",
        iteration_count=0
    )

    # Invoke Graph
    try:
        result = chat_graph.invoke(state)
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
