import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from agent.utils.logger import get_logger

logger = get_logger(__name__)

# We use the X-API-Key header to authorize requests
API_KEY_NAME = "X-API-Key"
api_key_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Security(api_key_scheme)) -> str:
    """
    Validates the provided API key against the environment variable.
    If MAAT_API_KEY is not set in the environment, it defaults to a secure dummy key for local testing.
    """
    expected_api_key = os.getenv("MAAT_API_KEY", "maat-local-dev-key-777")
    
    if api_key == expected_api_key:
        return api_key
    else:
        logger.warning(f"Authentication failed: invalid API key from request")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Could not validate API credentials. Received: {api_key}"
        )

