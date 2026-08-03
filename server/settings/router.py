
"""
Settings router for Ma'at Legal AI.

Provides REST endpoints for user preferences and settings management.
"""

from fastapi import APIRouter, Depends, status

from server.auth.dependencies import get_current_active_user
from server.common.logging import get_logger
from server.db.models import User, UserSettings
from server.db.models import AVAILABLE_CHAT_MODELS, AVAILABLE_EMBEDDING_MODELS
from server.settings.models import (
    AvailableModelsResponse,
    ChangePasswordRequest,
    DeleteAccountRequest,
    UserSettingsResponse,
    UserSettingsUpdate,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get("", response_model=UserSettingsResponse)
async def get_settings(current_user: User = Depends(get_current_active_user)):
    """Get current user's settings."""
    settings = current_user.settings
    if not settings:
        # Create default settings
        settings = UserSettings(user_id=current_user.id)
        await settings.insert()
        current_user.settings = settings
        await current_user.save()

    return UserSettingsResponse(
        theme=settings.theme,
        preferred_chat_model=settings.preferred_chat_model,
        preferred_embedding_model=settings.preferred_embedding_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        language=settings.language,
        notifications_enabled=settings.notifications_enabled,
    )


@router.patch("", response_model=UserSettingsResponse)
async def update_settings(
    request: UserSettingsUpdate,
    current_user: User = Depends(get_current_active_user),
):
    """Update user settings (partial update)."""
    settings = current_user.settings
    if not settings:
        settings = UserSettings(user_id=current_user.id)
        await settings.insert()
        current_user.settings = settings

    # Apply partial updates
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(settings, field, value)

    await settings.save()

    logger.info("User settings updated", user_id=str(current_user.id), fields=list(update_data.keys()))

    return UserSettingsResponse(
        theme=settings.theme,
        preferred_chat_model=settings.preferred_chat_model,
        preferred_embedding_model=settings.preferred_embedding_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        language=settings.language,
        notifications_enabled=settings.notifications_enabled,
    )


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get available LLM and embedding models."""
    return AvailableModelsResponse(
        chat_models=AVAILABLE_CHAT_MODELS,
        embedding_models=AVAILABLE_EMBEDDING_MODELS,
    )


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Change user password."""
    from server.auth.security import hash_password, verify_password

    # Verify current password
    if not verify_password(request.current_password, current_user.password_hash):
        from server.common.exceptions import AuthenticationError
        raise AuthenticationError("Current password is incorrect")

    # Hash and update new password
    current_user.password_hash = hash_password(request.new_password)
    await current_user.save()

    logger.info("Password changed", user_id=str(current_user.id))


@router.delete("/account", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    request: DeleteAccountRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Delete user account (requires password confirmation)."""
    from server.auth.security import verify_password

    # Verify password
    if not verify_password(request.password, current_user.password_hash):
        from server.common.exceptions import AuthenticationError
        raise AuthenticationError("Password is incorrect")

    # Delete all user data
    from server.db.models import ChatSession, ChatMessage

    # Delete messages
    await ChatMessage.find({"user_id": current_user.id}).delete()

    # Delete sessions
    await ChatSession.find({"user_id": current_user.id}).delete()

    # Delete settings
    if current_user.settings:
        await current_user.settings.delete()

    # Delete user
    await current_user.delete()

    logger.info("Account deleted", user_id=str(current_user.id))
