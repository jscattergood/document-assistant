"""
Document Assistant Backend
Main FastAPI application for document analysis and AI-powered assistance.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from src.api.documents import router as documents_router
from src.api.chat import router as chat_router
from src.api.confluence import router as confluence_router
from src.api.models import router as models_router
from src.document_processor.service import DocumentService

# Load environment variables
load_dotenv()

# Global service instance
document_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global document_service
    
    # Startup
    print("Starting Document Assistant...")
    document_service = DocumentService()
    # Initialize with MPNet model by default (best for semantic search)
    await document_service.initialize(embedding_model="mpnet", use_gpu=None)
    print("Document Assistant initialized successfully!")
    
    yield
    
    # Shutdown
    if document_service:
        await document_service.cleanup()
    print("Document Assistant shutdown complete.")

# Create FastAPI app
app = FastAPI(
    title="Document Assistant API",
    description="AI-powered document analysis and generation API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(confluence_router, prefix="/api/confluence", tags=["confluence"])
app.include_router(models_router, prefix="/api/models", tags=["models"])

# Mount static files for uploaded documents
if not os.path.exists("../data/documents"):
    os.makedirs("../data/documents", exist_ok=True)

app.mount("/data", StaticFiles(directory="../data"), name="data")

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Document Assistant API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "document-assistant"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 