"""
MongoDB connection management for Ma'at Legal AI.

Provides async MongoDB client via Motor and Beanie ODM initialization.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from beanie import init_beanie

from server.common.config import settings
from server.db.models import User, ChatSession, ChatMessage, UserSettings
from server.common.logging import get_logger

logger = get_logger(__name__)

# Global database client and database instances
_MONGO_CLIENT: Optional[AsyncIOMotorClient] = None
_DATABASE: Optional[AsyncIOMotorDatabase] = None


async def init_database() -> AsyncIOMotorDatabase:
    """
    Initialize MongoDB connection and Beanie ODM.

    Returns:
        AsyncIOMotorDatabase: The initialized database instance.

    Raises:
        RuntimeError: If database initialization fails.
    """
    global _MONGO_CLIENT, _DATABASE

    if _DATABASE is not None:
        logger.warning("Database already initialized, returning existing connection")
        return _DATABASE

    mongodb_uri = settings.MONGODB_URI
    db_name = settings.MONGODB_DB_NAME

    logger.info(f"Initializing MongoDB connection to {mongodb_uri}/{db_name}")

    try:
        # Create Motor client with connection pooling
        _MONGO_CLIENT = AsyncIOMotorClient(
            mongodb_uri,
            maxPoolSize=50,
            minPoolSize=5,
            maxIdleTimeMS=30000,
            waitQueueTimeoutMS=5000,
            serverSelectionTimeoutMS=10000,
        )

        _DATABASE = _MONGO_CLIENT[db_name]

        # Test the connection
        await _MONGO_CLIENT.admin.command("ping")
        logger.info("MongoDB connection established successfully")

        # Initialize Beanie with document models
        await init_beanie(
            database=_DATABASE,
            document_models=[User, ChatSession, ChatMessage, UserSettings],
        )
        logger.info("Beanie ODM initialized with document models")

        return _DATABASE

    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}")
        await close_database()
        raise RuntimeError(f"Database initialization failed: {e}") from e


async def close_database() -> None:
    """Close MongoDB connection and clean up resources."""
    global _MONGO_CLIENT, _DATABASE

    if _MONGO_CLIENT is not None:
        _MONGO_CLIENT.close()
        _MONGO_CLIENT = None
        _DATABASE = None
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """
    Get the initialized database instance.

    Returns:
        AsyncIOMotorDatabase: The database instance.

    Raises:
        RuntimeError: If database has not been initialized.
    """
    if _DATABASE is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _DATABASE


@asynccontextmanager
async def database_lifespan():
    """
    Context manager for database lifecycle.

    Usage:
        async with database_lifespan():
            # Database operations here
            pass
        # Database automatically closed
    """
    try:
        await init_database()
        yield get_database()
    finally:
        await close_database()


async def get_database_dependency() -> AsyncIOMotorDatabase:
    """
    FastAPI dependency for database injection.

    Returns:
        AsyncIOMotorDatabase: The database instance.
    """
    return get_database()


async def init_beanie_models(database: AsyncIOMotorDatabase) -> None:
    """
    Initialize Beanie ODM with document models on an existing database.

    Args:
        database: Existing database instance to initialize Beanie on.
    """
    await init_beanie(
        database=database,
        document_models=[User, ChatSession, ChatMessage, UserSettings],
    )
    logger.info("Beanie ODM initialized with document models on existing database")
