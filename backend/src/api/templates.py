"""
API endpoints for template management.
"""
import json
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..models.template import (
    Template, CreateTemplateRequest, UpdateTemplateRequest, 
    SyncTemplateRequest, TemplateResponse, TemplateListResponse
)
from ..document_processor.service import DocumentService

router = APIRouter()

# Helper function to convert HTML to markdown
def _convert_html_to_markdown(html_content: str) -> str:
    """Convert HTML content to basic markdown format."""
    if not html_content:
        return ""
    
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Convert HTML to basic markdown
    content = html_content
    
    # Convert headers
    content = content.replace('<h1>', '# ').replace('</h1>', '\n\n')
    content = content.replace('<h2>', '## ').replace('</h2>', '\n\n')
    content = content.replace('<h3>', '### ').replace('</h3>', '\n\n')
    content = content.replace('<h4>', '#### ').replace('</h4>', '\n\n')
    content = content.replace('<h5>', '##### ').replace('</h5>', '\n\n')
    content = content.replace('<h6>', '###### ').replace('</h6>', '\n\n')
    
    # Convert paragraphs
    content = content.replace('<p>', '').replace('</p>', '\n\n')
    
    # Convert bold and italic
    content = content.replace('<strong>', '**').replace('</strong>', '**')
    content = content.replace('<b>', '**').replace('</b>', '**')
    content = content.replace('<em>', '*').replace('</em>', '*')
    content = content.replace('<i>', '*').replace('</i>', '*')
    
    # Convert lists
    content = content.replace('<ul>', '').replace('</ul>', '\n')
    content = content.replace('<ol>', '').replace('</ol>', '\n')
    content = content.replace('<li>', '- ').replace('</li>', '\n')
    
    # Convert line breaks
    content = content.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    
    # Remove remaining HTML tags and get clean text
    soup = BeautifulSoup(content, 'html.parser')
    clean_content = soup.get_text()
    
    # Clean up excessive newlines
    lines = clean_content.split('\n')
    cleaned_lines = []
    prev_empty = False
    
    for line in lines:
        line = line.strip()
        if line == '':
            if not prev_empty:
                cleaned_lines.append('')
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    
    return '\n'.join(cleaned_lines).strip()

# Helper function to create auth headers
def _get_auth_headers(credentials) -> dict:
    """Create authentication headers for Confluence API requests."""
    if credentials.auth_type == "pat":
        return {
            "Authorization": f"Bearer {credentials.api_token}",
            "Content-Type": "application/json"
        }
    else:  # basic auth
        import base64
        auth_string = f"{credentials.username}:{credentials.api_token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        return {
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/json"
        }

# Dependency to get document service
def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    from main import document_service
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    return document_service


@router.get("/", response_model=TemplateListResponse)
async def list_templates(service: DocumentService = Depends(get_document_service)):
    """Get all templates."""
    try:
        templates = await service.list_templates()
        return TemplateListResponse(
            success=True,
            message=f"Retrieved {len(templates)} templates",
            templates=templates,
            count=len(templates)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing templates: {str(e)}")


@router.post("/", response_model=TemplateResponse)
async def create_template_from_url(
    request: CreateTemplateRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Create a template from a Confluence URL."""
    try:
        # Import confluence parsing functions
        from .confluence import (
            parse_confluence_url, fetch_confluence_page_info
        )
        
        # Parse the Confluence URL
        url_info = parse_confluence_url(request.source_url)
        auth_headers = _get_auth_headers(request.credentials)
        
        # Fetch content from Confluence
        page_info = await fetch_confluence_page_info(url_info, auth_headers)
        
        # Convert HTML to markdown
        content = _convert_html_to_markdown(page_info['content'])
        
        # Extract sections from the content
        sections = service._extract_sections_from_content(content)
        
        # Create template object
        template = Template(
            id="",  # Will be generated by save_template
            name=request.name or page_info['title'],
            description=request.description or f"Template based on: {page_info['title']}",
            source_url=request.source_url,
            content=content,
            sections=sections,
            template_type="confluence",
            space_key=page_info.get('space_key'),
            page_id=page_info.get('page_id'),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_synced=datetime.now(),
            sync_enabled=request.sync_enabled
        )
        
        # Save the template
        template_id = await service.save_template(template)
        saved_template = await service.get_template(template_id)
        
        return TemplateResponse(
            success=True,
            message=f"Template created successfully from {request.source_url}",
            template=saved_template
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating template: {str(e)}")


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get a template by ID."""
    try:
        template = await service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return TemplateResponse(
            success=True,
            message="Template retrieved successfully",
            template=template
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving template: {str(e)}")


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    request: UpdateTemplateRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Update an existing template."""
    try:
        template = await service.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Update fields if provided
        if request.name is not None:
            template.name = request.name
        if request.description is not None:
            template.description = request.description
        if request.content is not None:
            template.content = request.content
            # Re-extract sections from updated content
            template.sections = service._extract_sections_from_content(request.content)
        if request.sections is not None:
            template.sections = request.sections
        if request.sync_enabled is not None:
            template.sync_enabled = request.sync_enabled
        
        # Save the updated template
        await service.save_template(template)
        
        return TemplateResponse(
            success=True,
            message="Template updated successfully",
            template=template
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating template: {str(e)}")


@router.delete("/{template_id}")
async def delete_template(
    template_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Delete a template by ID."""
    try:
        success = await service.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "success": True,
            "message": "Template deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting template: {str(e)}")


@router.post("/{template_id}/sync", response_model=TemplateResponse)
async def sync_template(
    template_id: str,
    request: SyncTemplateRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Sync a template with its Confluence source."""
    try:
        template = await service.sync_template_from_confluence(template_id, request.credentials)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found or has no sync source")
        
        return TemplateResponse(
            success=True,
            message="Template synced successfully",
            template=template
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing template: {str(e)}")


@router.post("/sync-all")
async def sync_all_templates(
    request: SyncTemplateRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Sync all templates that have Confluence sources."""
    try:
        templates = await service.list_templates()
        syncable_templates = [t for t in templates if t.source_url and t.sync_enabled]
        
        if not syncable_templates:
            return {
                "success": True,
                "message": "No templates available for syncing",
                "synced_count": 0,
                "total_templates": len(templates)
            }
        
        synced_count = 0
        errors = []
        
        for template in syncable_templates:
            try:
                synced_template = await service.sync_template_from_confluence(
                    template.id, request.credentials
                )
                if synced_template:
                    synced_count += 1
            except Exception as e:
                errors.append(f"Error syncing template '{template.name}': {str(e)}")
        
        message = f"Synced {synced_count} of {len(syncable_templates)} templates"
        if errors:
            message += f". {len(errors)} errors occurred."
        
        return {
            "success": True,
            "message": message,
            "synced_count": synced_count,
            "total_syncable": len(syncable_templates),
            "total_templates": len(templates),
            "errors": errors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing templates: {str(e)}")


class BuiltinTemplate(BaseModel):
    """Model for built-in template information."""
    id: str
    name: str
    description: str
    sections: List[str]
    template_type: str = "builtin"


@router.get("/builtin/list")
async def get_builtin_templates():
    """Get list of built-in template types for content generation."""
    builtin_templates = [
        BuiltinTemplate(
            id="documentation",
            name="Technical Documentation",
            description="Comprehensive technical documentation with overview, requirements, implementation details",
            sections=["Overview", "Requirements", "Architecture", "Implementation", "Testing", "Deployment"]
        ),
        BuiltinTemplate(
            id="meeting_notes",
            name="Meeting Notes",
            description="Structured meeting notes with agenda, discussion points, and action items",
            sections=["Meeting Details", "Attendees", "Agenda", "Discussion", "Decisions", "Action Items"]
        ),
        BuiltinTemplate(
            id="project_plan",
            name="Project Plan",
            description="Detailed project planning document with timeline and milestones",
            sections=["Project Overview", "Objectives", "Scope", "Timeline", "Resources", "Risks", "Success Criteria"]
        ),
        BuiltinTemplate(
            id="knowledge_base",
            name="Knowledge Base Article",
            description="Educational content with step-by-step instructions and examples",
            sections=["Summary", "Prerequisites", "Step-by-Step Guide", "Examples", "Troubleshooting", "Related Topics"]
        ),
        BuiltinTemplate(
            id="tutorial",
            name="Tutorial/How-To",
            description="Tutorial content with hands-on instructions and code examples",
            sections=["Introduction", "Prerequisites", "Setup", "Tutorial Steps", "Code Examples", "Next Steps"]
        ),
        BuiltinTemplate(
            id="custom",
            name="Custom Template",
            description="Flexible template with custom sections",
            sections=[]
        )
    ]
    
    return {
        "success": True,
        "message": "Built-in templates retrieved successfully",
        "templates": builtin_templates,
        "count": len(builtin_templates)
    } 