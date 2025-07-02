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
        # Perform the query with source tracking
        response, sources = await service.query_documents_with_sources(
            query=request.message,
            document_ids=request.document_ids
        )
        
        return ChatResponse(
            success=True,
            message="Query processed successfully",
            response=response,
            sources=sources
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
        
        # Chat with documents with source tracking
        response, sources = await service.chat_with_documents_with_sources(
            message=request.message,
            conversation_history=conversation_history
        )
        
        return ChatResponse(
            success=True,
            message="Chat response generated successfully",
            response=response,
            sources=sources
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
        
        response, sources = await service.query_documents_with_sources(
            query=generation_prompt,
            document_ids=request.document_ids
        )
        
        return ChatResponse(
            success=True,
            message="Content generated successfully",
            response=response,
            sources=sources
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
        
        response, sources = await service.query_documents_with_sources(
            query=confluence_prompt,
            document_ids=request.document_ids
        )
        
        return {
            "success": True,
            "message": "Confluence draft generated successfully",
            "draft": response,
            "title": request.message,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Confluence draft: {str(e)}")

@router.post("/confluence-content-advanced")
async def generate_advanced_confluence_content(
    request: ChatRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Generate advanced Confluence content with enhanced AI prompting."""
    try:
        # Extract content type and requirements from the message
        content_prompt = f"""
        Create comprehensive Confluence content for: {request.message}
        
        Please analyze the request and determine the most appropriate content structure, then generate:
        
        1. **Content Analysis**: Identify the type of content needed (documentation, tutorial, meeting notes, etc.)
        2. **Structure Planning**: Create an optimal section hierarchy
        3. **Content Generation**: Produce well-formatted content using Confluence storage format
        
        Requirements:
        - Use proper Confluence markup (headers, lists, tables, code blocks, info panels)
        - Include relevant information from the knowledge base
        - Create comprehensive, professional documentation
        - Add appropriate cross-references and links
        - Use info/tip/warning panels for important information
        - Structure content logically with clear navigation
        
        Content Request: {request.message}
        
        Generate the content in Confluence storage format with proper HTML-like markup.
        """
        
        # Add conversation history if provided
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        else:
            conversation_history = []
        
        # Chat with documents using enhanced prompt with source tracking
        response, sources = await service.chat_with_documents_with_sources(
            message=content_prompt,
            conversation_history=conversation_history
        )
        
        return {
            "success": True,
            "message": "Advanced Confluence content generated successfully",
            "content": response,
            "content_type": "confluence_advanced",
            "format": "storage",
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating advanced Confluence content: {str(e)}")

@router.post("/confluence-template-content")
async def generate_template_based_content(
    request: dict,
    service: DocumentService = Depends(get_document_service)
):
    """Generate Confluence content based on specific templates and requirements."""
    try:
        topic = request.get('topic', '')
        template_type = request.get('template_type', 'general')
        additional_requirements = request.get('requirements', '')
        document_ids = request.get('document_ids', [])
        
        # Template-specific prompts
        template_prompts = {
            "api_documentation": f"""
            Create comprehensive API documentation for: {topic}
            
            Structure:
            1. API Overview and Introduction
            2. Authentication and Security
            3. Base URLs and Endpoints
            4. Request/Response Formats
            5. Endpoint Documentation (with examples)
            6. Error Handling and Status Codes
            7. Rate Limiting and Best Practices
            8. SDK and Code Examples
            9. Testing and Troubleshooting
            
            Use Confluence code blocks, tables, and info panels.
            """,
            
            "user_guide": f"""
            Create a comprehensive user guide for: {topic}
            
            Structure:
            1. Getting Started Overview
            2. Prerequisites and Setup
            3. Basic Operations (step-by-step)
            4. Advanced Features
            5. Common Use Cases and Examples
            6. Troubleshooting and FAQ
            7. Tips and Best Practices
            8. Additional Resources
            
            Use screenshots placeholders, step-by-step instructions, and tip panels.
            """,
            
            "process_documentation": f"""
            Create detailed process documentation for: {topic}
            
            Structure:
            1. Process Overview and Purpose
            2. Roles and Responsibilities
            3. Prerequisites and Requirements
            4. Detailed Process Steps
            5. Decision Points and Workflows
            6. Quality Checkpoints
            7. Exception Handling
            8. Related Processes and References
            
            Use flowcharts placeholders, tables, and warning panels for critical steps.
            """,
            
            "project_retrospective": f"""
            Create a project retrospective document for: {topic}
            
            Structure:
            1. Project Summary and Context
            2. Goals and Objectives Review
            3. What Went Well (Successes)
            4. What Could Be Improved (Challenges)
            5. Key Learnings and Insights
            6. Action Items for Future Projects
            7. Metrics and Performance Analysis
            8. Recommendations and Next Steps
            
            Use tables for action items, info panels for key insights.
            """,
            
            "release_notes": f"""
            Create comprehensive release notes for: {topic}
            
            Structure:
            1. Release Overview and Highlights
            2. New Features and Enhancements
            3. Bug Fixes and Improvements
            4. Breaking Changes and Migration Guide
            5. Performance and Security Updates
            6. Known Issues and Limitations
            7. Installation and Upgrade Instructions
            8. Support and Feedback Information
            
            Use tables for features, warning panels for breaking changes.
            """
        }
        
        base_prompt = template_prompts.get(template_type, f"""
        Create well-structured documentation for: {topic}
        
        Generate comprehensive, professional content with:
        - Clear headings and organization
        - Relevant examples and use cases
        - Best practices and recommendations
        - Proper Confluence formatting
        """)
        
        full_prompt = f"""
        {base_prompt}
        
        Additional Requirements: {additional_requirements}
        
        Please use information from the knowledge base to create accurate, comprehensive content.
        Format using Confluence storage format with proper markup.
        """
        
        response = await service.query_documents(
            query=full_prompt,
            document_ids=document_ids
        )
        
        return {
            "success": True,
            "message": f"Template-based content generated for {template_type}",
            "content": response,
            "template_type": template_type,
            "topic": topic
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating template-based content: {str(e)}")

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