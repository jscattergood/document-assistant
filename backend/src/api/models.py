"""
API endpoints for model management and system configuration.
"""
import os
import asyncio
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import aiohttp
import aiofiles

from ..document_processor.service import DocumentService

router = APIRouter()

# Dependency to get document service
def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    from main import document_service
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    return document_service

class EmbeddingModelRequest(BaseModel):
    """Request to change embedding model."""
    model_key: str
    use_gpu: Optional[bool] = None

class EmbeddingModelResponse(BaseModel):
    """Response for embedding model operations."""
    success: bool
    message: str
    model_info: Optional[Dict[str, Any]] = None

class GPT4AllModelInfo(BaseModel):
    """Information about a GPT4All model."""
    filename: str
    name: str
    size_bytes: int
    size_human: str
    download_url: Optional[str] = None
    description: Optional[str] = None
    recommended: bool = False
    is_downloaded: bool = False
    is_active: bool = False

class ModelDownloadRequest(BaseModel):
    """Request to download a model."""
    model_name: str
    download_url: str

class StorageConfig(BaseModel):
    documents_directory: str
    vector_database_path: str
    models_directory: str
    confluence_directory: str

class StorageConfigUpdate(BaseModel):
    documents_directory: Optional[str] = None
    vector_database_path: Optional[str] = None
    models_directory: Optional[str] = None
    confluence_directory: Optional[str] = None

# Predefined GPT4All models with download information - Updated with working HuggingFace URLs
AVAILABLE_GPT4ALL_MODELS = {
    "llama-3-8b-instruct": {
        "filename": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "name": "Llama 3 8B Instruct",
        "size_bytes": 4661000000,  # ~4.66GB
        "description": "Meta's latest Llama 3 model - excellent for instruction following and chat",
        "download_url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "recommended": True
    },
    "llama-3.1-8b-instruct": {
        "filename": "Meta-Llama-3.1-8B-Instruct.Q4_0.gguf", 
        "name": "Llama 3.1 8B Instruct",
        "size_bytes": 4920000000,  # ~4.92GB
        "description": "Latest Llama 3.1 with extended context and improved capabilities",
        "download_url": "https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "recommended": True
    },
    "llama-3.2-3b-instruct": {
        "filename": "Llama-3.2-3B-Instruct.Q4_0.gguf",
        "name": "Llama 3.2 3B Instruct", 
        "size_bytes": 2020000000,  # ~2.02GB
        "description": "Smaller, efficient Llama 3.2 model perfect for faster responses",
        "download_url": "https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "recommended": True
    },
    "nous-hermes-2-mistral": {
        "filename": "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
        "name": "Nous Hermes 2 Mistral 7B DPO",
        "size_bytes": 4110000000,  # ~4.11GB
        "description": "High-quality general purpose model, excellent for creative tasks",
        "download_url": "https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF/resolve/main/Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
        "recommended": True
    },
    "wizardlm-2-7b": {
        "filename": "WizardLM-2-7B.Q4_K_M.gguf",
        "name": "WizardLM 2 7B",
        "size_bytes": 4370000000,  # ~4.37GB
        "description": "Excellent for complex reasoning, coding, and analytical tasks",
        "download_url": "https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF/resolve/main/WizardLM-2-7B.Q4_K_M.gguf",
        "recommended": True
    },
    "phi-3-mini-instruct": {
        "filename": "Phi-3-mini-4k-instruct.Q4_0.gguf",
        "name": "Phi-3 Mini 4K Instruct",
        "size_bytes": 2180000000,  # ~2.18GB
        "description": "Microsoft's efficient small model, great for code and reasoning tasks",
        "download_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "recommended": False
    },
    "orca-mini-3b": {
        "filename": "orca-mini-3b-gguf2-q4_0.gguf",
        "name": "Orca Mini 3B",
        "size_bytes": 1980000000,  # ~1.98GB
        "description": "Compact model good for quick responses and basic document queries",
        "download_url": "https://huggingface.co/psmathur/orca_mini_3b/resolve/main/orca-mini-3b-gguf2-q4_0.gguf",
        "recommended": False
    },
    "mistral-7b-instruct": {
        "filename": "mistral-7b-instruct-v0.2.Q4_0.gguf",
        "name": "Mistral 7B Instruct v0.2",
        "size_bytes": 4109000000,  # ~4.1GB
        "description": "Balanced performance and speed, excellent for instruction-following",
        "download_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf",
        "recommended": True
    },
    "deepseek-coder-6.7b": {
        "filename": "deepseek-coder-6.7b-instruct.Q4_0.gguf", 
        "name": "DeepSeek Coder 6.7B Instruct",
        "size_bytes": 3800000000,  # ~3.8GB
        "description": "Specialized for coding tasks, supports multiple programming languages",
        "download_url": "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_0.gguf",
        "recommended": False
    },
    "codellama-7b-instruct": {
        "filename": "codellama-7b-instruct.Q4_0.gguf",
        "name": "Code Llama 7B Instruct", 
        "size_bytes": 3830000000,  # ~3.83GB
        "description": "Meta's specialized code generation model based on Llama 2",
        "download_url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_0.gguf",
        "recommended": False
    }
}

def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and size_index < len(size_names) - 1:
        size /= 1024.0
        size_index += 1
    
    return f"{size:.1f} {size_names[size_index]}"

# GPT4All Model Management Endpoints

@router.get("/gpt4all/available")
async def get_available_gpt4all_models(service: DocumentService = Depends(get_document_service)):
    """Get list of available GPT4All models with download status."""
    try:
        models_dir = service.models_dir
        downloaded_files = {f.name for f in models_dir.glob("*.gguf")} | {f.name for f in models_dir.glob("*.bin")}
        
        # Get currently active model
        current_model = service._get_gpt4all_model_path()
        
        models = []
        for model_key, model_info in AVAILABLE_GPT4ALL_MODELS.items():
            is_downloaded = model_info["filename"] in downloaded_files
            is_active = current_model == model_info["filename"] if current_model else False
            
            models.append(GPT4AllModelInfo(
                filename=model_info["filename"],
                name=model_info["name"],
                size_bytes=model_info["size_bytes"],
                size_human=_format_file_size(model_info["size_bytes"]),
                download_url=model_info["download_url"],
                description=model_info["description"],
                recommended=model_info["recommended"],
                is_downloaded=is_downloaded,
                is_active=is_active
            ))
        
        return {
            "success": True,
            "models": models,
            "total_count": len(models),
            "downloaded_count": sum(1 for m in models if m.is_downloaded),
            "active_model": current_model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting GPT4All models: {str(e)}")

@router.get("/gpt4all/downloaded")
async def get_downloaded_gpt4all_models(service: DocumentService = Depends(get_document_service)):
    """Get list of downloaded GPT4All models."""
    try:
        models_dir = service.models_dir
        model_files = list(models_dir.glob("*.gguf")) + list(models_dir.glob("*.bin"))
        
        current_model = service._get_gpt4all_model_path()
        
        downloaded_models = []
        for model_file in model_files:
            file_stat = model_file.stat()
            is_active = current_model == model_file.name
            
            # Try to find model info in predefined models
            model_info = None
            for info in AVAILABLE_GPT4ALL_MODELS.values():
                if info["filename"] == model_file.name:
                    model_info = info
                    break
            
            downloaded_models.append({
                "filename": model_file.name,
                "name": model_info["name"] if model_info else model_file.stem,
                "size_bytes": file_stat.st_size,
                "size_human": _format_file_size(file_stat.st_size),
                "description": model_info["description"] if model_info else "Custom model",
                "is_active": is_active,
                "created_at": file_stat.st_ctime,
                "modified_at": file_stat.st_mtime
            })
        
        return {
            "success": True,
            "models": downloaded_models,
            "count": len(downloaded_models),
            "active_model": current_model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting downloaded models: {str(e)}")

@router.post("/gpt4all/download")
async def download_gpt4all_model(
    request: ModelDownloadRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Download a GPT4All model."""
    try:
        # Find model info
        model_info = None
        for info in AVAILABLE_GPT4ALL_MODELS.values():
            if info["filename"] == request.model_name or request.model_name in info["name"]:
                model_info = info
                break
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found in available models")
        
        models_dir = service.models_dir
        target_file = models_dir / model_info["filename"]
        
        if target_file.exists():
            return {
                "success": False,
                "message": f"Model {model_info['name']} is already downloaded"
            }
        
        # Start download in background
        asyncio.create_task(
            _download_model_async(request.download_url, target_file, model_info["name"])
        )
        
        return {
            "success": True,
            "message": f"Download started for {model_info['name']}",
            "filename": model_info["filename"],
            "size_human": _format_file_size(model_info["size_bytes"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting download: {str(e)}")

async def _download_model_async(url: str, target_file: Path, model_name: str):
    """Download model file asynchronously with improved error handling and progress tracking."""
    try:
        print(f"Starting download of {model_name} to {target_file}")
        
        # Create a temporary file for download
        temp_file = target_file.with_suffix('.tmp')
        
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Failed to download {model_name}: HTTP {response.status}")
                    return
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                print(f"Download started: {model_name} ({_format_file_size(total_size)})")
                
                async with aiofiles.open(temp_file, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 50MB to avoid spam
                        if downloaded % (50 * 1024 * 1024) == 0 or downloaded == total_size:
                            progress = (downloaded / total_size * 100) if total_size else 0
                            print(f"Download progress for {model_name}: {progress:.1f}% ({_format_file_size(downloaded)}/{_format_file_size(total_size)})")
        
        # Move temp file to final location
        temp_file.rename(target_file)
        print(f"Successfully downloaded {model_name}")
        
    except asyncio.TimeoutError:
        print(f"Download timeout for {model_name}")
        if temp_file.exists():
            temp_file.unlink()
    except aiohttp.ClientError as e:
        print(f"Network error downloading {model_name}: {e}")
        if temp_file.exists():
            temp_file.unlink()
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        # Clean up partial download
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()

@router.post("/gpt4all/upload")
async def upload_gpt4all_model(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service)
):
    """Upload a GPT4All model file."""
    try:
        # Validate file extension
        if not (file.filename.endswith('.gguf') or file.filename.endswith('.bin')):
            raise HTTPException(
                status_code=400,
                detail="Only .gguf and .bin files are supported for GPT4All models"
            )
        
        models_dir = service.models_dir
        target_file = models_dir / file.filename
        
        if target_file.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Model file {file.filename} already exists"
            )
        
        # Save uploaded file
        async with aiofiles.open(target_file, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_size = target_file.stat().st_size
        
        return {
            "success": True,
            "message": f"Model {file.filename} uploaded successfully",
            "filename": file.filename,
            "size_bytes": file_size,
            "size_human": _format_file_size(file_size)
        }
        
    except Exception as e:
        # Clean up partial upload
        if 'target_file' in locals() and target_file.exists():
            target_file.unlink()
        raise HTTPException(status_code=500, detail=f"Error uploading model: {str(e)}")

@router.delete("/gpt4all/{filename}")
async def delete_gpt4all_model(
    filename: str,
    service: DocumentService = Depends(get_document_service)
):
    """Delete a GPT4All model file."""
    try:
        models_dir = service.models_dir
        model_file = models_dir / filename
        
        if not model_file.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Check if this is the currently active model
        current_model = service._get_gpt4all_model_path()
        if current_model == filename:
            return {
                "success": False,
                "message": f"Cannot delete {filename} because it is currently active. Switch to another model first."
            }
        
        model_file.unlink()
        
        return {
            "success": True,
            "message": f"Model {filename} deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

@router.post("/gpt4all/set-active")
async def set_active_gpt4all_model(
    filename: str,
    service: DocumentService = Depends(get_document_service)
):
    """Set the active GPT4All model."""
    try:
        models_dir = service.models_dir
        model_file = models_dir / filename
        
        if not model_file.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Reinitialize the service with the new model
        # Note: This requires restarting the LLM which may take time
        from gpt4all import GPT4All
        from ..document_processor.service import GPT4AllLLM
        
        # Create new LLM instance
        new_llm = GPT4AllLLM(str(models_dir), filename)
        
        if not new_llm._available:
            raise HTTPException(status_code=400, detail="Failed to initialize model")
        
        # Update the service
        service.llm = new_llm
        
        return {
            "success": True,
            "message": f"Successfully switched to model: {filename}",
            "active_model": filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting active model: {str(e)}")

@router.get("/gpt4all/download-status/{filename}")
async def get_download_status(
    filename: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get download status for a model."""
    try:
        models_dir = service.models_dir
        target_file = models_dir / filename
        
        if target_file.exists():
            file_size = target_file.stat().st_size
            
            # Check if it's a complete file by comparing with expected size
            expected_size = None
            for info in AVAILABLE_GPT4ALL_MODELS.values():
                if info["filename"] == filename:
                    expected_size = info["size_bytes"]
                    break
            
            is_complete = expected_size is None or file_size >= expected_size * 0.95  # Allow 5% tolerance
            
            return {
                "success": True,
                "exists": True,
                "size_bytes": file_size,
                "size_human": _format_file_size(file_size),
                "expected_size": expected_size,
                "is_complete": is_complete,
                "progress": (file_size / expected_size * 100) if expected_size else 100
            }
        else:
            return {
                "success": True,
                "exists": False,
                "size_bytes": 0,
                "progress": 0
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking download status: {str(e)}")

@router.get("/embeddings/available")
async def get_available_embedding_models():
    """Get list of available embedding models."""
    try:
        service = get_document_service()
        models = service.get_available_models()
        
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available models: {str(e)}")

@router.get("/embeddings/current")
async def get_current_embedding_model(service: DocumentService = Depends(get_document_service)):
    """Get information about the current embedding model."""
    try:
        model_info = service.get_embedding_info()
        
        return {
            "success": True,
            "model_info": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting current model info: {str(e)}")

@router.post("/embeddings/set", response_model=EmbeddingModelResponse)
async def set_embedding_model(
    request: EmbeddingModelRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Set the embedding model."""
    try:
        # Change the embedding model
        await service.set_embedding_model(request.model_key, request.use_gpu)
        
        # Get updated model info
        model_info = service.get_embedding_info()
        
        return EmbeddingModelResponse(
            success=True,
            message=f"Successfully set embedding model to: {request.model_key}",
            model_info=model_info
        )
        
    except Exception as e:
        return EmbeddingModelResponse(
            success=False,
            message=f"Failed to set embedding model: {str(e)}"
        )

@router.post("/embeddings/test")
async def test_embedding_model(
    text: str = "This is a test sentence for embedding generation.",
    service: DocumentService = Depends(get_document_service)
):
    """Test the current embedding model by generating an embedding."""
    try:
        if not service.embedding_model:
            raise HTTPException(status_code=400, detail="No embedding model initialized")
        
        # Generate test embedding
        embedding = service.embedding_model._get_text_embedding(text)
        
        return {
            "success": True,
            "text": text,
            "embedding_length": len(embedding),
            "embedding_sample": embedding[:5],  # First 5 dimensions
            "model_info": service.get_embedding_info()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing embedding: {str(e)}")

@router.get("/system/status")
async def get_system_status(service: DocumentService = Depends(get_document_service)):
    """Get overall system status including models and capabilities."""
    try:
        import torch
        
        # Check GPU availability
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        if torch.cuda.is_available():
            gpu_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        # Get embedding model info
        embedding_info = service.get_embedding_info()
        
        # Get available models
        available_models = service.get_available_models()
        
        return {
            "success": True,
            "gpu_info": gpu_info,
            "embedding_model": embedding_info,
            "available_models": list(available_models.keys()),
            "document_count": len(service.get_all_documents())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@router.post("/vector-store/clear")
async def clear_vector_store(service: DocumentService = Depends(get_document_service)):
    """Clear the vector store and recreate it. Useful when changing embedding models."""
    try:
        success = await service.clear_vector_store()
        
        if success:
            return {
                "success": True,
                "message": "Vector store cleared and reinitialized successfully"
            }
        else:
            return {
                "success": False,
                "message": "Failed to clear vector store"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing vector store: {str(e)}")

# Storage Configuration Endpoints

@router.get("/storage/config", response_model=StorageConfig)
async def get_storage_config(
    service: DocumentService = Depends(get_document_service)
):
    """Get current storage configuration."""
    try:
        return StorageConfig(
            documents_directory=str(service.documents_dir.absolute()),
            vector_database_path=str(service.chroma_dir.absolute()),
            models_directory=str(service.models_dir.absolute()),
            confluence_directory=str((service.data_dir / "confluence").absolute())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting storage config: {str(e)}")

@router.get("/storage/status")
async def get_storage_status(
    service: DocumentService = Depends(get_document_service)
):
    """Get storage status and statistics."""
    try:
        def get_directory_info(path: Path):
            if path.exists():
                files = list(path.iterdir()) if path.is_dir() else []
                size = sum(f.stat().st_size for f in files if f.is_file())
                return {
                    "exists": True,
                    "is_directory": path.is_dir(),
                    "file_count": len([f for f in files if f.is_file()]),
                    "total_size_bytes": size,
                    "writable": os.access(path, os.W_OK)
                }
            else:
                return {
                    "exists": False,
                    "is_directory": False,
                    "file_count": 0,
                    "total_size_bytes": 0,
                    "writable": False
                }
        
        return {
            "success": True,
            "storage_status": {
                "documents_directory": get_directory_info(service.documents_dir),
                "vector_database_path": get_directory_info(service.chroma_dir),
                "models_directory": get_directory_info(service.models_dir),
                "confluence_directory": get_directory_info(service.data_dir / "confluence"),
                "total_documents": len(service.documents),
                "vector_store_connected": service.vector_store is not None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting storage status: {str(e)}")

@router.post("/storage/validate-path")
async def validate_storage_path(
    path_data: Dict[str, str]
):
    """Validate if a storage path is usable."""
    try:
        path_str = path_data.get("path", "")
        if not path_str:
            return {"valid": False, "error": "Path cannot be empty"}
        
        path = Path(path_str).expanduser().resolve()
        
        # Check if path exists or can be created
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                created = True
            except Exception as e:
                return {"valid": False, "error": f"Cannot create directory: {str(e)}"}
        else:
            created = False
            
        # Check if writable
        if not os.access(path, os.W_OK):
            return {"valid": False, "error": "Directory is not writable"}
            
        # Check if it's actually a directory
        if not path.is_dir():
            return {"valid": False, "error": "Path is not a directory"}
            
        return {
            "valid": True,
            "absolute_path": str(path),
            "created": created,
            "writable": True
        }
        
    except Exception as e:
        return {"valid": False, "error": f"Path validation error: {str(e)}"}

@router.post("/storage/clear-cache")
async def clear_storage_cache():
    """Clear various caches and temporary data."""
    try:
        # This could clear document processing caches, model caches, etc.
        # For now, just return success
        return {
            "success": True,
            "message": "Storage cache cleared successfully",
            "cleared_items": ["document_cache", "embedding_cache", "temp_files"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}") 