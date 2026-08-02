"""
Main FastAPI application for Ma'at Legal AI - API Gateway.

This is the API Gateway that routes to Auth, Chat, Settings, and Agent services.
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

from server.common.config import settings
from server.common.logging import get_logger, set_correlation_id
from server.db.connection import close_database, init_database

# Import routers from all services
from server.auth.router import router as auth_router
from server.chat.router import router as chat_router
from server.settings.router import router as settings_router
from server.api.routes import router as legacy_router  # Legacy chat endpoint

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown."""
    # Startup
    logger.info("Starting Ma'at Legal AI API Gateway...")
    await init_database()

    # Initialize LangSmith tracing if configured
    if settings.LANGSMITH_TRACING_V2 and settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        logger.info("LangSmith tracing enabled", project=settings.LANGSMITH_PROJECT)

    yield

    # Shutdown
    logger.info("Shutting down Ma'at Legal AI API Gateway...")
    await close_database()


app = FastAPI(
    title=settings.APP_NAME,
    description="Backend RAG API for the Indian Legal Code AI Assistant",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Correlation-ID"],
)


# Correlation ID Middleware
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Inject correlation ID into every request for distributed tracing."""
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    set_correlation_id(cid)
    logger.info(f"{request.method} {request.url.path} [cid={cid}]")
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} [cid={cid}]")
    return response


# Security Headers Middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    if settings.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Rate Limiting (optional - using slowapi if available)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled")
except ImportError:
    logger.warning("slowapi not installed - rate limiting disabled")


# Health Check Endpoints
@app.get("/health/live")
async def liveness_check():
    """Liveness probe - service is running."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe - service can handle requests."""
    checks = {"api": "ok"}
    try:
        db = await init_database()
        await db.command("ping")
        checks["mongodb"] = "ok"
    except Exception as e:
        checks["mongodb"] = f"error: {str(e)[:100]}"
        logger.warning("Readiness check failed: mongodb", error=str(e))

    # Check Pinecone (if configured)
    if settings.PINECONE_API_KEY:
        checks["pinecone"] = "configured"
    else:
        checks["pinecone"] = "not_configured"

    # Check NVIDIA NIM
    if settings.NVIDIA_NIM_KEY:
        checks["nvidia_nim"] = "configured"
    else:
        checks["nvidia_nim"] = "not_configured"

    return {"status": "ready", "checks": checks}


# Register Service Routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(settings_router, prefix="/api/v1")
# Legacy chat endpoint (for backward compatibility)
app.include_router(legacy_router, prefix="/api/v1")


# Serve static frontend files if they exist (for single-container Docker deployment)
static_dir = Path(__file__).parent.parent.parent / "app" / "dist"
if static_dir.exists():
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        path = static_dir / full_path
        if path.exists() and path.is_file():
            return FileResponse(path)
        return FileResponse(static_dir / "index.html")


logger.info("Ma'at Legal AI API Gateway initialized", version=settings.APP_VERSION)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
