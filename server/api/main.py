import os
import uuid
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router
from agent.utils.logger import get_logger, set_correlation_id

# Ensure .env is loaded
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

logger = get_logger(__name__)

# CORS configuration - use environment variable for production, fallback to localhost for dev
cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000")
allow_origins = [origin.strip() for origin in cors_origins.split(",")]

app = FastAPI(
    title="Ma-at Legal AI API",
    description="Backend RAG API for the Indian Legal Code AI Assistant.",
    version="1.0.0"
)

# CORS Middleware - restricted origins from env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Correlation-ID"],
)


# Correlation ID Middleware — tags every request with a unique trace ID

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Injects a correlation ID into every request for distributed tracing."""
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    set_correlation_id(cid)
    logger.info(f"{request.method} {request.url.path} [cid={cid}]")
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [cid={cid}]")
    return response


# Register routes
app.include_router(router)

# Serve static frontend files if they exist (for single-container Docker deployment)
static_dir = Path(__file__).parent.parent.parent / "app" / "dist"
if static_dir.exists():
    # Mount the assets directory explicitly
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Catch-all route to serve the SPA
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        path = static_dir / full_path
        if path.exists() and path.is_file():
            return FileResponse(path)
        return FileResponse(static_dir / "index.html")

logger.info("Ma-at Legal AI API initialized successfully.")

if __name__ == "__main__":
    import uvicorn
    # When run directly
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
