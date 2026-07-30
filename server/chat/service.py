"""
Chat service for Ma'at Legal AI.

Provides business logic for chat session and message management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from beanie import PydanticObjectId
from pydantic import ValidationError as PydanticValidationError

from server.chat.models import (
    ChatMessageBase,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    MessageRole,
    SessionListResponse,
)
from server.common.exceptions import NotFoundError, ResourceNotFoundError
from server.common.logging import get_logger
from server.db.models import ChatMessage, ChatSession, User

logger = get_logger(__name__)


class ChatService:
    """Service for chat session and message operations."""

    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100

    async def create_session(
        self,
        user_id: PydanticObjectId,
        request: ChatSessionCreate,
    ) -> ChatSessionResponse:
        """
        Create a new chat session for a user.

        Args:
            user_id: User ID
            request: Session creation request

        Returns:
            ChatSessionResponse: Created session
        """
        # Verify user exists
        user = await User.get(user_id)
        if not user:
            logger.warning("Session creation failed: User not found", user_id=str(user_id))
            raise ResourceNotFoundError("User", str(user_id))

        session = ChatSession(
            user_id=user_id,
            title=request.title or "New Conversation",
            is_archived=request.is_archived,
            memory_summary="",
            message_count=0,
        )
        await session.insert()

        logger.info("Chat session created", session_id=str(session.id), user_id=str(user_id))
        return self._session_to_response(session)

    async def get_session(
        self,
        session_id: str,
        user_id: PydanticObjectId,
    ) -> ChatSessionResponse:
        """
        Get a chat session by ID.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)

        Returns:
            ChatSessionResponse: Session data

        Raises:
            ResourceNotFoundError: If session not found or not owned by user
        """
        try:
            session_oid = PydanticObjectId(session_id)
        except PydanticValidationError:
            raise ResourceNotFoundError("ChatSession", session_id)

        session = await ChatSession.get(session_oid)
        if not session:
            raise ResourceNotFoundError("ChatSession", session_id)

        # Verify ownership
        if session.user_id != user_id:
            logger.warning(
                "Session access denied: User does not own session",
                session_id=session_id,
                user_id=str(user_id),
                owner_id=str(session.user_id),
            )
            raise ResourceNotFoundError("ChatSession", session_id)

        return self._session_to_response(session)

    async def list_sessions(
        self,
        user_id: PydanticObjectId,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        include_archived: bool = False,
    ) -> SessionListResponse:
        """
        List user's chat sessions with pagination.

        Args:
            user_id: User ID
            page: Page number (1-indexed)
            page_size: Items per page
            include_archived: Include archived sessions

        Returns:
            SessionListResponse: Paginated session list
        """
        page_size = min(page_size, self.MAX_PAGE_SIZE)
        skip = (page - 1) * page_size

        # Build query
        query = {"user_id": user_id}
        if not include_archived:
            query["is_archived"] = False

        # Get total count
        total = await ChatSession.find(query).count()

        # Get paginated sessions
        sessions = await ChatSession.find(query).sort(-ChatSession.updated_at).skip(skip).limit(page_size).to_list()

        items = [self._session_to_response(s) for s in sessions]

        return SessionListResponse(
            sessions=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    async def update_session(
        self,
        session_id: str,
        user_id: PydanticObjectId,
        request: ChatSessionUpdate,
    ) -> ChatSessionResponse:
        """
        Update a chat session.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)
            request: Update request

        Returns:
            ChatSessionResponse: Updated session

        Raises:
            ResourceNotFoundError: If session not found or not owned by user
        """
        try:
            session_oid = PydanticObjectId(session_id)
        except PydanticValidationError:
            raise ResourceNotFoundError("ChatSession", session_id)

        session = await ChatSession.get(session_oid)
        if not session:
            raise ResourceNotFoundError("ChatSession", session_id)

        if session.user_id != user_id:
            raise ResourceNotFoundError("ChatSession", session_id)

        # Apply updates
        if request.title is not None:
            session.title = request.title
        if request.is_archived is not None:
            session.is_archived = request.is_archived

        session.updated_at = datetime.utcnow()
        await session.save()

        logger.info("Session updated", session_id=session_id, user_id=str(user_id))
        return self._session_to_response(session)

    async def delete_session(
        self,
        session_id: str,
        user_id: PydanticObjectId,
    ) -> None:
        """
        Delete a chat session and all its messages.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)

        Raises:
            ResourceNotFoundError: If session not found or not owned by user
        """
        try:
            session_oid = PydanticObjectId(session_id)
        except PydanticValidationError:
            raise ResourceNotFoundError("ChatSession", session_id)

        session = await ChatSession.get(session_oid)
        if not session:
            raise ResourceNotFoundError("ChatSession", session_id)

        if session.user_id != user_id:
            raise ResourceNotFoundError("ChatSession", session_id)

        # Delete all messages in session
        await ChatMessage.find({"session_id": session_oid}).delete()

        # Delete session
        await session.delete()

        logger.info("Chat session deleted", session_id=session_id, user_id=str(user_id))

    async def add_message(
        self,
        session_id: str,
        user_id: PydanticObjectId,
        request: ChatMessageCreate,
    ) -> ChatMessageResponse:
        """
        Add a message to a chat session.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)
            request: Message creation request

        Returns:
            ChatMessageResponse: Created message

        Raises:
            ResourceNotFoundError: If session not found or not owned by user
        """
        try:
            session_oid = PydanticObjectId(session_id)
        except PydanticValidationError:
            raise ResourceNotFoundError("ChatSession", session_id)

        session = await ChatSession.get(session_oid)
        if not session:
            raise ResourceNotFoundError("ChatSession", session_id)

        if session.user_id != user_id:
            raise ResourceNotFoundError("ChatSession", session_id)

        message = ChatMessage(
            session_id=session_oid,
            user_id=user_id,
            role=request.role,
            content=request.content,
            law_domain=request.law_domain,
            metadata=request.metadata,
        )
        await message.insert()

        # Update session message count and timestamp
        session.message_count += 1
        session.updated_at = datetime.utcnow()
        await session.save()

        logger.debug("Message added to session", session_id=session_id, message_id=str(message.id))
        return self._message_to_response(message)

    async def get_history(
        self,
        session_id: str,
        user_id: PydanticObjectId,
        limit: Optional[int] = None,
    ) -> List[ChatMessageResponse]:
        """
        Get chat history for a session.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)
            limit: Optional limit on number of messages (most recent)

        Returns:
            List[ChatMessageResponse]: Messages in chronological order

        Raises:
            ResourceNotFoundError: If session not found or not owned by user
        """
        try:
            session_oid = PydanticObjectId(session_id)
        except PydanticValidationError:
            raise ResourceNotFoundError("ChatSession", session_id)

        session = await ChatSession.get(session_oid)
        if not session:
            raise ResourceNotFoundError("ChatSession", session_id)

        if session.user_id != user_id:
            raise ResourceNotFoundError("ChatSession", session_id)

        query = ChatMessage.find({"session_id": session_oid}).sort(ChatMessage.created_at)
        if limit:
            query = query.limit(limit)

        messages = await query.to_list()
        return [self._message_to_response(m) for m in messages]

    async def update_memory_summary(
        self,
        session_id: str,
        user_id: PydanticObjectId,
        memory_summary: str,
    ) -> ChatSessionResponse:
        """
        Update session memory summary.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)
            memory_summary: New memory summary

        Returns:
            ChatSessionResponse: Updated session

        Raises:
            ResourceNotFoundError: If session not found or not owned by user
        """
        try:
            session_oid = PydanticObjectId(session_id)
        except PydanticValidationError:
            raise ResourceNotFoundError("ChatSession", session_id)

        session = await ChatSession.get(session_oid)
        if not session:
            raise ResourceNotFoundError("ChatSession", session_id)

        if session.user_id != user_id:
            raise ResourceNotFoundError("ChatSession", session_id)

        session.memory_summary = memory_summary
        session.updated_at = datetime.utcnow()
        await session.save()

        logger.info("Memory summary updated", session_id=session_id, user_id=str(user_id))
        return self._session_to_response(session)

    @staticmethod
    def _session_to_response(session: ChatSession) -> ChatSessionResponse:
        """Convert ChatSession document to response model."""
        return ChatSessionResponse(
            id=str(session.id),
            user_id=str(session.user_id),
            title=session.title,
            memory_summary=session.memory_summary,
            message_count=session.message_count,
            is_archived=session.is_archived,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    @staticmethod
    def _message_to_response(message: ChatMessage) -> ChatMessageResponse:
        """Convert ChatMessage document to response model."""
        return ChatMessageResponse(
            id=str(message.id),
            session_id=str(message.session_id),
            user_id=str(message.user_id),
            role=message.role,
            content=message.content,
            law_domain=message.law_domain,
            metadata=message.metadata,
            created_at=message.created_at,
        )