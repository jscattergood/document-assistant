"""
API endpoints for chat functionality.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
import uuid
import asyncio
from datetime import datetime

from ..models.document import ChatRequest, ChatResponse, ChatMessage
from ..document_processor.service import DocumentService

router = APIRouter()

# Job storage (in production, use Redis or a proper database)
jobs: Dict[str, Dict[str, Any]] = {}

# Dependency to get document service
def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    from main import document_service
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    return document_service

# Store conversation history (in production, use a proper database)
conversations = {}

class BackgroundChatRequest(BaseModel):
    """Request model for background chat processing."""
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    document_ids: Optional[List[str]] = None

class JobResponse(BaseModel):
    """Response model for background job creation."""
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    """Response model for job status checking."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: Optional[str] = None
    response: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

async def process_chat_background(
    job_id: str,
    message: str,
    conversation_history: Optional[List[Dict[str, str]]],
    document_ids: Optional[List[str]],
    service: DocumentService
):
    """Background task to process chat request."""
    try:
        # Update job status to processing
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["updated_at"] = datetime.now()
        
        # Process the chat request
        response = await service.chat_with_documents(
            message=message,
            conversation_history=conversation_history
        )
        
        # Update job with success
        jobs[job_id].update({
            "status": "completed",
            "response": response,
            "completed_at": datetime.now(),
            "updated_at": datetime.now()
        })
        
    except Exception as e:
        # Update job with error
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(),
            "updated_at": datetime.now()
        })

@router.post("/chat-background", response_model=JobResponse)
async def start_background_chat(
    request: BackgroundChatRequest,
    background_tasks: BackgroundTasks,
    service: DocumentService = Depends(get_document_service)
):
    """Start a background chat processing job."""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job record
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "message": "Chat request queued for processing",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "request": {
                "message": request.message,
                "conversation_history": request.conversation_history,
                "document_ids": request.document_ids
            }
        }
        
        # Add background task
        background_tasks.add_task(
            process_chat_background,
            job_id,
            request.message,
            request.conversation_history,
            request.document_ids,
            service
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Chat request queued for background processing"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting background chat: {str(e)}")

@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a background job."""
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            message=job.get("message"),
            response=job.get("response"),
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            error=job.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")

@router.delete("/job/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up a completed job."""
    try:
        if job_id in jobs:
            del jobs[job_id]
            return {"success": True, "message": "Job cleaned up successfully"}
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up job: {str(e)}")

@router.post("/query", response_model=ChatResponse)
async def query_documents(
    request: ChatRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Query documents with a natural language question."""
    try:
        # Perform the query
        response = await service.query_documents(
            query=request.message,
            document_ids=request.document_ids
        )
        
        return ChatResponse(
            success=True,
            message="Query processed successfully",
            response=response,
            sources=[]  # TODO: Extract source documents from response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    request: ChatRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Have a conversation with documents."""
    try:
        # Convert conversation history to the format expected by the service
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        # Chat with documents
        response = await service.chat_with_documents(
            message=request.message,
            conversation_history=conversation_history
        )
        
        return ChatResponse(
            success=True,
            message="Chat response generated successfully",
            response=response,
            sources=[]  # TODO: Extract source documents from response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@router.post("/generate", response_model=ChatResponse)
async def generate_document_content(
    request: ChatRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Generate new document content based on existing documents."""
    try:
        # Create a specialized prompt for document generation
        generation_prompt = f"""
        Based on the documents in the knowledge base, please help generate content for: {request.message}
        
        Please provide well-structured, comprehensive content that:
        1. Uses information and patterns from the existing documents
        2. Maintains consistency with the style and tone of the source materials
        3. Is properly formatted using markdown for better readability
        4. Includes relevant details and examples where appropriate
        
        Use markdown formatting:
        - **Bold** for important points and headings
        - Bullet points or numbered lists for organization
        - `Code blocks` for technical terms or examples
        - ## Headers to organize sections
        - > Blockquotes for important notes
        
        Request: {request.message}
        """
        
        response = await service.query_documents(
            query=generation_prompt,
            document_ids=request.document_ids
        )
        
        return ChatResponse(
            success=True,
            message="Content generated successfully",
            response=response,
            sources=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

@router.post("/confluence-draft")
async def generate_confluence_draft(
    request: ChatRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Generate a Confluence page draft based on documents."""
    try:
        # Create a specialized prompt for Confluence page generation
        confluence_prompt = f"""
        Please create a Confluence page draft for: {request.message}
        
        Format the response as a Confluence page with:
        1. A clear title
        2. Proper headings and sections
        3. Bullet points or numbered lists where appropriate
        4. Tables if relevant data is available
        5. Links and references to source materials
        
        Use information from the knowledge base to create comprehensive, well-organized content.
        Include proper Confluence markup formatting.
        
        Page topic: {request.message}
        """
        
        response = await service.query_documents(
            query=confluence_prompt,
            document_ids=request.document_ids
        )
        
        return {
            "success": True,
            "message": "Confluence draft generated successfully",
            "draft": response,
            "title": request.message,
            "sources": []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Confluence draft: {str(e)}")

@router.get("/history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history by ID."""
    try:
        history = conversations.get(conversation_id, [])
        return {
            "success": True,
            "conversation_id": conversation_id,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@router.delete("/history/{conversation_id}")
async def clear_conversation_history(conversation_id: str):
    """Clear conversation history."""
    try:
        if conversation_id in conversations:
            del conversations[conversation_id]
        
        return {
            "success": True,
            "message": "Conversation history cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

class DocumentAnalysisRequest(BaseModel):
    document_ids: List[str]
    analysis_type: str  # "summary", "comparison", "key_insights"

@router.post("/analyze")
async def analyze_documents(
    request: DocumentAnalysisRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Perform analysis on selected documents."""
    try:
        if not request.document_ids:
            raise HTTPException(status_code=400, detail="No documents selected for analysis")
        
        # Get document titles for context
        document_titles = []
        for doc_id in request.document_ids:
            doc = service.get_document(doc_id)
            if doc:
                document_titles.append(doc.title)
        
        # Create analysis prompt based on type
        if request.analysis_type == "summary":
            prompt = f"""Please provide a comprehensive summary of the following documents: {', '.join(document_titles)}

Format your response using markdown:
- Use **bold** for key points and document names
- Use bullet points for main findings
- Use ## headers to organize sections
- Use > blockquotes for important insights"""
        elif request.analysis_type == "comparison":
            prompt = f"""Please compare and contrast the following documents, highlighting similarities and differences: {', '.join(document_titles)}

Format your response using markdown:
- Use **bold** for document names and key differences
- Use bullet points or tables for comparisons
- Use ## headers for "Similarities" and "Differences" sections
- Use > blockquotes for key insights"""
        elif request.analysis_type == "key_insights":
            prompt = f"""Please extract the key insights and important information from the following documents: {', '.join(document_titles)}

Format your response using markdown:
- Use **bold** for key insights and important findings
- Use numbered lists for prioritized insights
- Use ## headers to organize by topic or document
- Use > blockquotes for critical information"""
        else:
            prompt = f"""Please analyze the following documents: {', '.join(document_titles)}

Format your response using markdown:
- Use **bold** for important findings
- Use bullet points for organized analysis
- Use ## headers to structure your analysis
- Use > blockquotes for key takeaways"""
        
        response = await service.query_documents(
            query=prompt,
            document_ids=request.document_ids
        )
        
        return {
            "success": True,
            "analysis_type": request.analysis_type,
            "documents_analyzed": document_titles,
            "analysis": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing documents: {str(e)}") 