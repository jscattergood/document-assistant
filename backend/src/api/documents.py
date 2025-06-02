"""
API endpoints for document management.
"""
import os
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..models.document import DocumentResponse, Document, DocumentSummary
from ..document_processor.service import DocumentService

router = APIRouter()

# Dependency to get document service
def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    from main import document_service
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    return document_service

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service)
):
    """Upload and process a document."""
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm"}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        upload_dir = "../data/documents"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        # Handle duplicate filenames
        counter = 1
        base_name, extension = os.path.splitext(file.filename)
        while os.path.exists(file_path):
            new_filename = f"{base_name}_{counter}{extension}"
            file_path = os.path.join(upload_dir, new_filename)
            counter += 1
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        document = await service.process_uploaded_file(file_path, os.path.basename(file_path))
        
        return DocumentResponse(
            success=True,
            message="Document uploaded and processed successfully",
            document=document
        )
        
    except Exception as e:
        # Clean up file if processing failed
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/", response_model=DocumentResponse)
async def list_documents(service: DocumentService = Depends(get_document_service)):
    """Get list of all processed documents."""
    try:
        documents = service.get_all_documents()
        return DocumentResponse(
            success=True,
            message="Documents retrieved successfully",
            documents=documents
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get a specific document by ID."""
    try:
        document = service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse(
            success=True,
            message="Document retrieved successfully",
            document=document
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Delete a document."""
    try:
        success = await service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"success": True, "message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.get("/{document_id}/summary")
async def get_document_summary(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Generate a summary for a specific document."""
    try:
        document = service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Generate summary using the query engine
        summary_query = f"Please provide a comprehensive summary of the document titled '{document.title}'. Include key points and main topics."
        summary_response = await service.query_documents(summary_query, [document_id])
        
        # Extract key points (this is a simplified version)
        key_points_query = f"List the key points from the document titled '{document.title}' as bullet points."
        key_points_response = await service.query_documents(key_points_query, [document_id])
        
        # For now, we'll return a simplified summary structure
        # In a production system, you'd want more sophisticated parsing
        summary = DocumentSummary(
            id=document_id,
            title=document.title,
            summary=summary_response,
            key_points=key_points_response.split('\n') if key_points_response else [],
            topics=[]  # Could extract topics using NLP techniques
        )
        
        return {"success": True, "summary": summary}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@router.get("/{document_id}/content")
async def get_document_content(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get the full content of a document."""
    try:
        document = service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "document_id": document_id,
            "title": document.title,
            "content": document.content,
            "type": document.type,
            "metadata": document.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving content: {str(e)}") 