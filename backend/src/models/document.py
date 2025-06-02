"""
Pydantic models for document handling and API responses.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    CONFLUENCE = "confluence"

class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ERROR = "error"

class DocumentBase(BaseModel):
    """Base document model."""
    title: str = Field(..., description="Document title")
    type: DocumentType = Field(..., description="Document type")
    content: Optional[str] = Field(None, description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class DocumentCreate(DocumentBase):
    """Model for creating a new document."""
    pass

class DocumentUpdate(BaseModel):
    """Model for updating a document."""
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Document(DocumentBase):
    """Complete document model."""
    id: str = Field(..., description="Unique document ID")
    file_path: Optional[str] = Field(None, description="Path to the document file")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    
    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    """API response model for documents."""
    success: bool = True
    message: str = "Operation successful"
    document: Optional[Document] = None
    documents: Optional[List[Document]] = None

class DocumentSummary(BaseModel):
    """Document summary model."""
    id: str
    title: str
    summary: str
    key_points: List[str]
    topics: List[str]

class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[str]] = Field(default=None, description="Source documents")

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    document_ids: Optional[List[str]] = Field(default=None, description="Documents to query")
    conversation_history: Optional[List[ChatMessage]] = Field(default=None)

class ChatResponse(BaseModel):
    """Chat response model."""
    success: bool = True
    message: str = "Query processed successfully"
    response: str = Field(..., description="AI response")
    sources: List[str] = Field(default_factory=list, description="Source documents")
    conversation_id: Optional[str] = None

class ConfluenceConfig(BaseModel):
    """Confluence configuration model."""
    url: str = Field(..., description="Confluence instance URL")
    username: str = Field(..., description="Confluence username")
    api_token: str = Field(..., description="Confluence API token")
    space_key: Optional[str] = Field(None, description="Default space key")

class ConfluencePage(BaseModel):
    """Confluence page model."""
    id: str
    title: str
    content: str
    space_key: str
    url: str
    created_at: datetime
    updated_at: datetime 