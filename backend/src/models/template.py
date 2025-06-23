"""
Template models for the document assistant.
"""
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ConfluenceCredentials(BaseModel):
    """Confluence credentials for template operations."""
    url: str
    username: Optional[str] = None
    api_token: str
    auth_type: str = "pat"


class Template(BaseModel):
    """Template model for storing and managing content templates."""
    id: str
    name: str
    description: str
    source_url: Optional[str] = None  # Confluence URL if imported
    content: str  # Markdown content
    sections: List[str] = Field(default_factory=list)
    template_type: str = "custom"  # custom, builtin, confluence
    space_key: Optional[str] = None
    page_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_synced: Optional[datetime] = None
    sync_enabled: bool = True


class CreateTemplateRequest(BaseModel):
    """Request model for creating a template from a Confluence URL."""
    source_url: str
    credentials: ConfluenceCredentials
    name: Optional[str] = None
    description: Optional[str] = None
    sync_enabled: bool = True


class UpdateTemplateRequest(BaseModel):
    """Request model for updating an existing template."""
    name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    sections: Optional[List[str]] = None
    sync_enabled: Optional[bool] = None


class SyncTemplateRequest(BaseModel):
    """Request model for syncing a template with its Confluence source."""
    credentials: ConfluenceCredentials


class TemplateResponse(BaseModel):
    """Response model for template operations."""
    success: bool
    message: str
    template: Optional[Template] = None


class TemplateListResponse(BaseModel):
    """Response model for listing templates."""
    success: bool
    message: str
    templates: List[Template]
    count: int 