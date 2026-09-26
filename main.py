import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.routes import router as api_router

# Initialize FastAPI application
app = FastAPI(
    title="DocuFlow - Semantic Embedding & Retrieval Engine",
    description="A high-performance semantic search, document ingestion, and RAG conversational pipeline.",
    version="1.0.0"
)

# Enable CORS for local development and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router at /api
app.include_router(api_router, prefix="/api", tags=["Semantic API"])

# Directory containing frontend static assets
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
index_file = os.path.join(frontend_dir, "index.html")

if os.path.exists(frontend_dir):
    # Mount static assets (css, js, images)
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    # Serve the Top-Tier UI directly on root "/" as well as "/ui" and "/app"
    @app.get("/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    @app.get("/app", include_in_schema=False)
    async def serve_ui():
        return FileResponse(index_file)

# Dedicated Health Check endpoint
@app.get("/health", tags=["System"])
async def root_health():
    from api.routes import search_engine
    stats = search_engine.get_stats()
    return {
        "status": "healthy",
        "service": "DocuFlow Engine",
        "ui": "http://localhost:8000/",
        "api_docs": "http://localhost:8000/docs",
        "total_indexed_chunks": stats.get("total_chunks", 0),
        "model": stats.get("model_name", "all-MiniLM-L6-v2")
    }


if __name__ == "__main__":
    host = os.getenv("DOCUFLOW_HOST", "0.0.0.0")
    port = int(os.getenv("DOCUFLOW_PORT", 8000))
    print(f"\n========================================================")
    print(f"🚀 DocuFlow Server starting at http://{host}:{port}")
    print(f"🎨 Top-Tier Frontend UI: http://localhost:{port}/")
    print(f"📚 API Documentation:   http://localhost:{port}/docs")
    print(f"========================================================\n")
    uvicorn.run("main:app", host=host, port=port, reload=True)
