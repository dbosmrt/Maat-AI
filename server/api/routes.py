"""Legacy chat routes for Ma'at Legal AI.

Provides backward-compatible endpoints for the chat flow.
This will be used by the frontend until migrated to the new /api/v1/chats endpoints.
"""

import os
import time
import uuid
from typing import List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, status
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.messages.utils import get_buffer_string

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

from server.api.models import ChatRequest, ChatResponse, StartSessionResponse, ChatHistoryResponse, SessionItem, SessionListResponse
from server.agent.chat_graph import get_chat_graph
from server.agent.state import AgentState
from server.agent.utils.logger import get_logger
from server.db.connection import get_database
from server.db.models import User, PydanticObjectId, ChatMessageCreate, MessageRole
from server.chat.service import ChatService
from server.auth.dependencies import get_current_active_user

logger = get_logger(__name__)

# Token limit for chat history (leave room for prompt + response)
_MAX_HISTORY_TOKENS = 3000
_ENCODING = "cl100k_base"  # GPT-4 / cl100k_base encoding


def _count_tokens(messages: List[BaseMessage]) -> int:
    """Count tokens in a list of messages using tiktoken if available, else estimate."""
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
    """Truncate message history to fit within token limit (sliding window from end)."""
    if not messages:
        return []

    count = _count_tokens(messages)
    if count <= max_tokens:
        return messages

    # Sliding window: keep most recent messages
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


chat_service = ChatService()

router = APIRouter()


@router.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "Ma'at Legal AI API is running"}


@router.post("/api/v1/chat/start", response_model=StartSessionResponse)
def start_chat(current_user: User = Depends(get_current_active_user)):
    """Initializes a new chat session (legacy endpoint - use /api/v1/chats)."""
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"New chat session created: {session_id} for user {current_user.id}")
    return StartSessionResponse(
        session_id=session_id,
        message="Session started successfully. Use /api/v1/chats for new session management."
    )


@router.get("/api/v1/chat/sessions", response_model=SessionListResponse)
def list_sessions(current_user: User = Depends(get_current_active_user)):
    """Lists all available chat sessions (legacy endpoint - use /api/v1/chats)."""
    # Note: This is legacy; sessions are now in MongoDB
    # Return empty for now - frontend should use /api/v1/chats
    return SessionListResponse(sessions=[])


@router.delete("/api/v1/chat/{session_id}")
def delete_session(session_id: str, current_user: User = Depends(get_current_active_user)):
    """Deletes a chat session (legacy endpoint)."""
    logger.warning(f"Legacy delete session called: {session_id}")
    raise HTTPException(status_code=410, detail="Use /api/v1/chats/{session_id} for session management")


@router.get("/api/v1/chat/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str, current_user: User = Depends(get_current_active_user)):
    """Retrieves the message history for a given session (legacy endpoint)."""
    try:
        user_id = PydanticObjectId(current_user.id)
        messages = await chat_service.get_history(session_id, user_id)
        history = [{"type": msg.role.value, "content": msg.content} for msg in messages]
        return ChatHistoryResponse(session_id=session_id, history=history)
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/api/v1/chat/{session_id}", response_model=ChatResponse)
async def invoke_chat(
    session_id: str,
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Core RAG inference endpoint (legacy - use /api/v1/chats for full features)."""
    start_time = time.perf_counter()

    if session_id.lower() == "new":
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"Auto-created new session: {session_id} for user {current_user.id}")
    else:
        logger.info(f"Chat invoked on session {session_id}: '{request.query[:80]}...'")

    # Get chat history from MongoDB
    user_id = PydanticObjectId(current_user.id)

    try:
        history = await chat_service.get_history(session_id, user_id)
    except Exception:
        history = []

    # Convert to LangChain messages
    lc_history: List[Any] = []
    for msg in history:
        if msg.role.value == "human":
            lc_history.append(HumanMessage(content=msg.content))
        elif msg.role.value == "ai":
            lc_history.append(AIMessage(content=msg.content))

    # Append the new user query
    lc_history.append(HumanMessage(content=request.query))

    # Token-aware truncation
    max_history_tokens = int(os.environ.get("MAX_HISTORY_TOKENS", "4000"))
    lc_history = _truncate_history_to_token_limit(lc_history, max_history_tokens)

    # Get user's preferred model settings
    user_settings = current_user.settings
    model_config = {}
    if user_settings:
        model_config = {
            "user_chat_model": user_settings.preferred_chat_model,
            "user_temperature": user_settings.temperature,
            "user_top_p": user_settings.top_p,
            "user_max_tokens": user_settings.max_tokens,
        }

    # Build the initial state for the graph
    state: AgentState = {
        "session_id": session_id,
        "chat_history": lc_history,
        "memory_summary": "",  # Could be loaded from session
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
        **model_config  # Flatten user settings into state
    }

    # Use pre-compiled graph (singleton)
    graph = get_chat_graph()

    # Invoke Graph with user settings
    try:
        result = graph.invoke(state)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Pipeline failed after {elapsed_ms:.0f}ms for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    generation = result.get("generation", "Failed to generate response.")
    domain = result.get("law_domain", "General")

    # Save the new history to MongoDB
    # Save user message
    await chat_service.add_message(
        session_id, user_id,
        ChatMessageCreate(role=MessageRole.HUMAN, content=request.query)
    )

    # Save AI response
    await chat_service.add_message(
        session_id, user_id,
        ChatMessageCreate(
            role=MessageRole.AI,
            content=generation,
            law_domain=domain
        )
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Chat completed for session {session_id} in {elapsed_ms:.0f}ms | domain={domain}")

    return ChatResponse(
        session_id=session_id,
        generation=generation,
        law_domain=domain
    )
