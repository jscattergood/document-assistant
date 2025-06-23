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
from src.api.templates import router as templates_router
from src.api.web_import import router as web_import_router
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
    
    # Check if we should auto-start Ollama
    try:
        import json
        from pathlib import Path
        
        settings_file = Path("../data/app_settings.json")
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                auto_start_ollama = settings.get('auto_start_ollama', False)
                
                if auto_start_ollama:
                    print("Auto-starting Ollama...")
                    await _auto_start_ollama()
    except Exception as e:
        print(f"Error checking auto-start settings: {e}")
    
    print("Document Assistant initialized successfully!")
    
    yield
    
    # Shutdown
    if document_service:
        await document_service.cleanup()
    print("Document Assistant shutdown complete.")

async def _auto_start_ollama():
    """Helper function to auto-start Ollama on server startup."""
    try:
        import subprocess
        import platform
        import time
        import requests
        
        # Check if Ollama is already running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                print("Ollama is already running")
                return
        except:
            pass
        
        print("Attempting to start Ollama...")
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            try:
                subprocess.run(["brew", "services", "start", "ollama"], 
                             check=True, capture_output=True, timeout=10)
                print("Started Ollama using brew services")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                try:
                    subprocess.Popen(["ollama", "serve"], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    print("Started Ollama using direct execution")
                except FileNotFoundError:
                    print("Ollama not found - please install Ollama")
                    return
        
        elif system == "linux":
            try:
                subprocess.run(["systemctl", "--user", "start", "ollama"], 
                             check=True, capture_output=True, timeout=10)
                print("Started Ollama using systemctl")
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                try:
                    subprocess.Popen(["ollama", "serve"], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    print("Started Ollama using direct execution")
                except FileNotFoundError:
                    print("Ollama not found - please install Ollama")
                    return
        
        # Wait and verify startup (give it more time on startup)
        print("Waiting for Ollama to start...")
        for i in range(10):  # Wait up to 10 seconds
            time.sleep(1)
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=3)
                if response.status_code == 200:
                    print("Ollama started successfully!")
                    return
            except:
                continue
        
        print("Ollama started but may still be initializing...")
        
    except Exception as e:
        print(f"Error auto-starting Ollama: {e}")

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
app.include_router(templates_router, prefix="/api/templates", tags=["templates"])
app.include_router(web_import_router, prefix="/api", tags=["web-import"])

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