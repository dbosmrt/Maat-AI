"""
Chat router for Ma'at Legal AI.

Provides REST endpoints for chat session and message management.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query, status

from server.auth.dependencies import get_current_active_user
from server.chat.models import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    SessionListResponse,
)
from server.chat.service import ChatService
from server.common.exceptions import ResourceNotFoundError
from server.common.logging import get_logger
from server.db.models import User

logger = get_logger(__name__)

router = APIRouter(prefix="/chats", tags=["Chat"])
chat_service = ChatService()


@router.post("", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: ChatSessionCreate,
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new chat session.

    - **title**: Optional session title (default: "New Conversation")
    """
    logger.info("Creating new chat session", user_id=str(current_user.id), title=request.title)
    return await chat_service.create_session(current_user.id, request)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    include_archived: bool = Query(False, description="Include archived sessions"),
    current_user: User = Depends(get_current_active_user),
):
    """
    List user's chat sessions with pagination.

    - **page**: Page number (1-indexed)
    - **page_size**: Items per page (max 100)
    - **include_archived**: Include archived sessions
    """
    return await chat_service.list_sessions(
        current_user.id,
        page=page,
        page_size=page_size,
        include_archived=include_archived,
    )


@router.get("/{session_id}", response_model=ChatSessionResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get a chat session by ID.

    - **session_id**: Session ID
    """
    return await chat_service.get_session(session_id, current_user.id)


@router.patch("/{session_id}", response_model=ChatSessionResponse)
async def update_session(
    session_id: str,
    request: ChatSessionUpdate,
    current_user: User = Depends(get_current_active_user),
):
    """
    Update a chat session.

    - **session_id**: Session ID
    - **title**: Optional new title
    - **is_archived**: Optional archive status
    """
    return await chat_service.update_session(session_id, current_user.id, request)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a chat session and all its messages.

    - **session_id**: Session ID
    """
    await chat_service.delete_session(session_id, current_user.id)


@router.get("/{session_id}/messages", response_model=list)
async def get_messages(
    session_id: str,
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Limit messages (most recent)"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get chat history for a session.

    - **session_id**: Session ID
    - **limit**: Optional limit on number of messages (most recent)
    """
    messages = await chat_service.get_history(session_id, current_user.id, limit=limit)
    # Convert to dict for response
    return [msg.model_dump(mode="json") for msg in messages]


@router.post("/{session_id}/messages/memory", response_model=ChatSessionResponse)
async def update_memory_summary(
    session_id: str,
    memory_summary: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Update session memory summary (used by AI agent for context).

    - **session_id**: Session ID
    - **memory_summary**: New memory summary
    """
    return await chat_service.update_memory_summary(session_id, current_user.id, memory_summary)