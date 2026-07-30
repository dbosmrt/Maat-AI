"""
Auth service for Ma'at Legal AI.

Provides authentication, authorization, and user management.
"""

from server.auth.dependencies import (
    get_current_active_user,
    get_current_user,
    get_current_user_id,
    get_optional_user,
    require_verified_user,
)
from server.auth.models import (
    AuthResponse,
    AvailableModelsResponse,
    LoginRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from server.auth.router import router
from server.auth.security import (
    create_access_token,
    create_password_reset_token,
    create_refresh_token,
    create_token_pair,
    decode_token,
    get_token_expiration,
    get_token_user_id,
    hash_password,
    is_token_expired,
    verify_password,
)

__all__ = [
    # Router
    "router",
    # Security
    "create_access_token",
    "create_refresh_token",
    "create_password_reset_token",
    "create_token_pair",
    "decode_token",
    "get_token_user_id",
    "get_token_expiration",
    "is_token_expired",
    "hash_password",
    "verify_password",
    # Dependencies
    "get_current_user_id",
    "get_current_user",
    "get_current_active_user",
    "get_optional_user",
    "require_verified_user",
    # Models
    "AuthResponse",
    "AvailableModelsResponse",
    "LoginRequest",
    "RefreshTokenRequest",
    "RegisterRequest",
    "TokenResponse",
    "UserResponse",
]