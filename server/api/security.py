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
    MAAT_API_KEY must be set in the environment; no default fallback.
    """
    expected_api_key = os.getenv("MAAT_API_KEY")

    if expected_api_key is None:
        logger.critical("MAAT_API_KEY environment variable is not set. Failing startup.")
        raise RuntimeError("MAAT_API_KEY environment variable is required but not set")

    if not api_key:
        logger.warning("Authentication failed: missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing API credentials"
        )

    if api_key == expected_api_key:
        return api_key

    logger.warning("Authentication failed: invalid API key from request")
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate API credentials"
    )
