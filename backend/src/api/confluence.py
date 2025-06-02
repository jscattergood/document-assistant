"""
API endpoints for Confluence integration.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import re
import uuid
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
import time

from ..models.document import (
    ConfluenceConfig, ConfluencePage, ConfluencePageSync, 
    ConfluencePageSyncRequest, ConfluencePageSyncResponse,
    ConfluenceTemporaryIngestRequest, ConfluenceTemporaryIngestResponse
)
from ..document_processor.service import DocumentService

router = APIRouter()

# Dependency to get document service
def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    from main import document_service
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    return document_service

# Storage for sync pages and temporary pages (in production, use database)
sync_pages_storage = {}
temporary_pages_storage = {}

# Remove configuration storage - all config should be client-side only for security

# Remove global credential storage - credentials will be passed with each request
# confluence_config = None

class ConfluenceCredentials(BaseModel):
    """Confluence credentials for individual requests."""
    url: str
    username: Optional[str] = None
    api_token: str
    auth_type: str = "pat"

def get_auth_headers(credentials: ConfluenceCredentials) -> dict:
    """Generate authentication headers from credentials."""
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    if credentials.auth_type == "pat":
        headers['Authorization'] = f'Bearer {credentials.api_token}'
    else:
        import base64
        auth_string = base64.b64encode(f"{credentials.username}:{credentials.api_token}".encode()).decode()
        headers['Authorization'] = f'Basic {auth_string}'
    
    return headers

class ConfluenceTestRequest(BaseModel):
    url: str
    username: Optional[str] = None  # Not required for PAT
    token: str
    space_key: Optional[str] = None
    auth_type: str = "pat"  # "pat" for Personal Access Token, "basic" for username/password

@router.post("/test")
async def test_confluence_connection(request: ConfluenceTestRequest):
    """Test Confluence connection with provided credentials."""
    try:
        print(f"Testing Confluence connection to: {request.url}")
        print(f"Auth type: {request.auth_type}")
        print(f"Username: {request.username}")
        print(f"Space key: {request.space_key}")
        
        # Import here to avoid circular imports
        import requests
        from requests.auth import HTTPBasicAuth
        
        # Clean URL (remove trailing slash)
        base_url = request.url.rstrip('/')
        print(f"Cleaned base URL: {base_url}")
        
        # Set up authentication based on type
        if request.auth_type == "pat":
            # For Personal Access Token, use Bearer authentication
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {request.token}'
            }
            auth = None
        else:
            # For username/password, use Basic authentication
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            auth = HTTPBasicAuth(request.username, request.token)
        
        # Try different API endpoints depending on Confluence type
        # For Confluence Cloud with SSO, try the v3 API first
        if 'atlassian.net' in base_url:
            test_url = f"{base_url}/wiki/rest/api/user/current"
        else:
            # For all other instances (including enterprise/SSO), use the standard API path
            test_url = f"{base_url}/rest/api/user/current"
            
        print(f"Testing URL: {test_url}")
        
        # Mask sensitive headers for logging
        safe_headers = {k: ('Bearer ***MASKED***' if k == 'Authorization' and v.startswith('Bearer ') else v) 
                       for k, v in headers.items()}
        print(f"Auth headers: {safe_headers}")
        
        response = requests.get(
            test_url,
            auth=auth,
            headers=headers,
            timeout=10,
            allow_redirects=False  # Don't follow redirects to login pages
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response content (first 200 chars): {response.text[:200]}")
        
        # Check for SSO redirects
        if response.status_code in [301, 302, 303, 307, 308]:
            location = response.headers.get('Location', '')
            if 'login' in location.lower() or 'auth' in location.lower() or 'microsoft' in location.lower():
                return {
                    "success": False,
                    "message": "This Confluence instance uses SSO authentication. API access with Personal Access Tokens may not be supported, or you may need to use a different authentication method."
                }
        
        # Check for HTML content (usually login pages)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            if 'sign in' in response.text.lower() or 'login' in response.text.lower():
                return {
                    "success": False,
                    "message": "Confluence is redirecting to a login page. This usually means: 1) The URL is incorrect, 2) SSO is required, or 3) API access is not enabled. Try using your Confluence Cloud URL (e.g., https://yourcompany.atlassian.net)."
                }
        
        if response.status_code == 200:
            try:
                user_data = response.json()
            except ValueError:
                return {
                    "success": False,
                    "message": "Connected to Confluence but received invalid response format. Please check your Confluence URL."
                }
            
            # If space_key is provided, test access to that space
            if request.space_key:
                space_url = f"{base_url}/rest/api/space/{request.space_key}"
                print(f"Testing space access: {space_url}")
                
                space_response = requests.get(
                    space_url, 
                    auth=auth, 
                    headers=headers,  # Use the same headers as the user test
                    timeout=10,
                    allow_redirects=False
                )
                
                print(f"Space response status: {space_response.status_code}")
                print(f"Space response headers: {dict(space_response.headers)}")
                print(f"Space response content (first 200 chars): {space_response.text[:200]}")
                
                if space_response.status_code == 200:
                    try:
                        space_data = space_response.json()
                        return {
                            "success": True,
                            "message": f"Successfully connected to Confluence! Access to space '{space_data.get('name', request.space_key)}' confirmed.",
                            "user": user_data.get('displayName', request.username or 'Unknown'),
                            "space": space_data.get('name')
                        }
                    except ValueError:
                        return {
                            "success": False,
                            "message": f"Connected to Confluence and found space '{request.space_key}', but received invalid response format."
                        }
                elif space_response.status_code == 404:
                    return {
                        "success": False,
                        "message": f"Connected to Confluence, but space '{request.space_key}' not found or not accessible."
                    }
                elif space_response.status_code == 403:
                    return {
                        "success": False,
                        "message": f"Connected to Confluence, but access to space '{request.space_key}' is forbidden. Check your permissions."
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Connected to Confluence, but cannot access space '{request.space_key}'. Status: {space_response.status_code}"
                    }
            else:
                return {
                    "success": True,
                    "message": f"Successfully connected to Confluence! Welcome, {user_data.get('displayName', request.username or 'Unknown')}.",
                    "user": user_data.get('displayName', request.username or 'Unknown')
                }
        elif response.status_code == 401:
            return {
                "success": False,
                "message": "Authentication failed. Please check your username and API token."
            }
        elif response.status_code == 403:
            return {
                "success": False,
                "message": "Access forbidden. Your account may not have sufficient permissions."
            }
        else:
            return {
                "success": False,
                "message": f"Connection failed with status code: {response.status_code}. Please check your Confluence URL."
            }
            
    except requests.exceptions.ConnectTimeout:
        return {
            "success": False,
            "message": "Connection timeout. Please check your Confluence URL and network connection."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": "Cannot connect to Confluence. Please verify the URL is correct and accessible."
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Request error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }

class ConfluenceImportRequest(BaseModel):
    space_key: str
    page_ids: Optional[List[str]] = None
    max_pages: Optional[int] = 10

@router.post("/import")
async def import_confluence_pages(
    request: ConfluenceImportRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Import pages from Confluence into the document index."""
    try:
        # Initialize Confluence reader
        from llama_index.readers.confluence import ConfluenceReader
        
        reader = ConfluenceReader(
            base_url=request.space_key,
            username=request.space_key,
            api_token=request.space_key
        )
        
        # Import pages
        if request.page_ids:
            # Import specific pages
            documents = reader.load_data(
                page_ids=request.page_ids,
                include_attachments=False
            )
        else:
            # Import from space
            documents = reader.load_data(
                space_key=request.space_key,
                include_attachments=False,
                max_num_results=request.max_pages
            )
        
        # Process imported documents
        imported_docs = []
        for doc in documents:
            try:
                # Create document record
                doc_id = f"confluence_{doc.metadata.get('page_id', 'unknown')}"
                
                # Process the document through our service
                # Note: This is a simplified version - you'd want to adapt the service
                # to handle Confluence documents more directly
                
                imported_docs.append({
                    "id": doc_id,
                    "title": doc.metadata.get('title', 'Untitled'),
                    "space_key": doc.metadata.get('space_key'),
                    "page_id": doc.metadata.get('page_id'),
                    "url": doc.metadata.get('url')
                })
                
            except Exception as e:
                print(f"Error processing Confluence document: {e}")
                continue
        
        return {
            "success": True,
            "message": f"Successfully imported {len(imported_docs)} pages from Confluence",
            "imported_pages": imported_docs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing from Confluence: {str(e)}")

class ConfluencePageRequest(BaseModel):
    space_key: str
    title: str
    content: str
    parent_page_id: Optional[str] = None

@router.post("/create-page")
async def create_confluence_page(request: ConfluencePageRequest):
    """Create a new page in Confluence."""
    try:
        # This would require implementing page creation via Confluence API
        # For now, return the content that would be created
        
        return {
            "success": True,
            "message": "Page content prepared (creation not implemented yet)",
            "page_data": {
                "space_key": request.space_key,
                "title": request.title,
                "content": request.content,
                "parent_page_id": request.parent_page_id
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating Confluence page: {str(e)}")

@router.get("/spaces")
async def list_confluence_spaces():
    """List available Confluence spaces."""
    try:
        return {
            "success": True,
            "spaces": [
                {
                    "key": request.space_key or "EXAMPLE",
                    "name": "Example Space",
                    "type": "global"
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing spaces: {str(e)}")

@router.get("/spaces/{space_key}/pages")
async def list_space_pages(space_key: str, limit: int = 25):
    """List pages in a Confluence space."""
    try:
        return {
            "success": True,
            "space_key": space_key,
            "pages": [
                {
                    "id": "123456",
                    "title": "Example Page",
                    "url": f"{request.space_key}/spaces/{space_key}/pages/123456"
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing pages: {str(e)}")

class ConfluenceSearchRequest(BaseModel):
    query: str
    space_key: Optional[str] = None

@router.post("/search")
async def search_confluence(
    request: ConfluenceSearchRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Search Confluence pages and integrate results with document query."""
    try:
        # First, search local Confluence documents
        search_query = f"Search Confluence content for: {request.query}"
        if request.space_key:
            search_query += f" in space {request.space_key}"
        
        local_results = await service.query_documents(search_query)
        
        return {
            "success": True,
            "query": request.query,
            "local_results": local_results,
            "confluence_results": []  # Would implement live search here
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Confluence: {str(e)}")

def parse_confluence_url(web_url: str) -> dict:
    """Parse Confluence web URL to extract page ID, space key, and generate API URL."""
    try:
        parsed = urlparse(web_url)
        page_id = None
        space_key = None
        page_title = None
        
        # Handle different Confluence URL formats
        if 'pageId=' in web_url:
            # Format: https://wiki.autodesk.com/spaces/viewspace.action?key=~scattej&pageId=123456
            query_params = parse_qs(parsed.query)
            page_id = query_params.get('pageId', [None])[0]
            space_key = query_params.get('key', [None])[0]
        elif '/display/' in web_url:
            # Format: https://wiki.autodesk.com/display/SPACE/Page+Title
            path_parts = parsed.path.split('/')
            if 'display' in path_parts:
                display_index = path_parts.index('display')
                if display_index + 1 < len(path_parts):
                    space_key = path_parts[display_index + 1]
                if display_index + 2 < len(path_parts):
                    page_title = path_parts[display_index + 2]
                    # URL decode the page title
                    from urllib.parse import unquote
                    page_title = unquote(page_title).replace('+', ' ')
        elif '/pages/' in web_url:
            # Format: https://wiki.autodesk.com/spaces/SPACE/pages/123456/Page+Title
            path_parts = parsed.path.split('/')
            if 'pages' in path_parts:
                page_index = path_parts.index('pages')
                if page_index + 1 < len(path_parts):
                    page_id = path_parts[page_index + 1]
                    # Try to find space key
                    if 'spaces' in path_parts:
                        space_index = path_parts.index('spaces')
                        if space_index + 1 < len(path_parts):
                            space_key = path_parts[space_index + 1]
        elif '/x/' in web_url:
            # Format: https://wiki.autodesk.com/x/ABC123 (short URLs)
            path_parts = parsed.path.split('/')
            if 'x' in path_parts:
                x_index = path_parts.index('x')
                if x_index + 1 < len(path_parts):
                    page_id = path_parts[x_index + 1]
                    space_key = None  # Will need to fetch from API
        else:
            raise ValueError("Unrecognized Confluence URL format")
        
        # Generate API URL base
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        if 'api.' in parsed.netloc:
            api_base = base_url
        else:
            api_base = base_url.replace('wiki.', 'api.wiki.')
        
        return {
            "page_id": page_id,
            "space_key": space_key,
            "page_title": page_title,
            "api_base": api_base,
            "base_url": base_url,
            "web_url": web_url
        }
        
    except Exception as e:
        raise ValueError(f"Error parsing Confluence URL: {str(e)}")

async def resolve_page_by_title(api_base: str, space_key: str, page_title: str, auth_headers: dict) -> dict:
    """Resolve a page title to page ID and API URL."""
    import requests
    
    try:
        # Search for the page by title and space
        search_url = f"{api_base}/rest/api/content"
        params = {
            'spaceKey': space_key,
            'title': page_title,
            'expand': 'space,body.storage,version'
        }
        
        response = requests.get(
            search_url,
            headers=auth_headers,
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if results:
                page = results[0]  # Take the first match
                page_id = page.get('id')
                api_url = f"{api_base}/rest/api/content/{page_id}"
                
                return {
                    "page_id": page_id,
                    "api_url": api_url,
                    "title": page.get('title', 'Unknown Title'),
                    "space_key": page.get('space', {}).get('key', 'Unknown'),
                    "content": page.get('body', {}).get('storage', {}).get('value', ''),
                    "version": page.get('version', {}).get('number', 1),
                    "created": page.get('history', {}).get('createdDate'),
                    "updated": page.get('version', {}).get('when')
                }
            else:
                raise Exception(f"Page '{page_title}' not found in space '{space_key}'")
        else:
            raise Exception(f"Search request failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        raise Exception(f"Error resolving page by title: {str(e)}")

async def fetch_confluence_page_info(url_info: dict, auth_headers: dict) -> dict:
    """Fetch page information from Confluence API."""
    import requests
    
    try:
        # If we have a page_id, use direct API call
        if url_info.get('page_id'):
            api_url = f"{url_info['api_base']}/rest/api/content/{url_info['page_id']}"
            response = requests.get(
                api_url,
                headers=auth_headers,
                params={'expand': 'space,body.storage,version'},
                timeout=10
            )
            
            if response.status_code == 200:
                page_data = response.json()
                return {
                    "title": page_data.get('title', 'Unknown Title'),
                    "space_key": page_data.get('space', {}).get('key', 'Unknown'),
                    "content": page_data.get('body', {}).get('storage', {}).get('value', ''),
                    "version": page_data.get('version', {}).get('number', 1),
                    "created": page_data.get('history', {}).get('createdDate'),
                    "updated": page_data.get('version', {}).get('when'),
                    "page_id": url_info['page_id'],
                    "api_url": api_url
                }
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        # If we have page_title, resolve it first
        elif url_info.get('page_title') and url_info.get('space_key'):
            page_info = await resolve_page_by_title(
                url_info['api_base'], 
                url_info['space_key'], 
                url_info['page_title'], 
                auth_headers
            )
            page_info['api_url'] = f"{url_info['api_base']}/rest/api/content/{page_info['page_id']}"
            return page_info
        
        else:
            raise Exception("Insufficient information to fetch page - need either page_id or (space_key + page_title)")
            
    except Exception as e:
        raise Exception(f"Error fetching page info: {str(e)}")

@router.post("/sync/add", response_model=ConfluencePageSyncResponse)
async def add_pages_to_sync(request: ConfluencePageSyncRequest):
    """Add Confluence pages to the sync list."""
    try:
        # Set up authentication headers
        auth_headers = get_auth_headers(request.credentials)
        
        synced_pages = []
        errors = []
        
        for web_url in request.web_urls:
            try:
                # Parse the URL
                url_info = parse_confluence_url(web_url)
                
                # Fetch page information
                page_info = await fetch_confluence_page_info(url_info, auth_headers)
                
                # Create sync record
                sync_id = str(uuid.uuid4())
                sync_page = ConfluencePageSync(
                    id=sync_id,
                    web_url=web_url,
                    page_id=page_info['page_id'],
                    space_key=page_info['space_key'],
                    title=page_info['title'],
                    api_url=page_info['api_url']
                )
                
                # Store in memory (in production, save to database)
                sync_pages_storage[sync_id] = sync_page
                synced_pages.append(sync_page)
                
            except Exception as e:
                errors.append(f"Error processing {web_url}: {str(e)}")
        
        return ConfluencePageSyncResponse(
            success=len(synced_pages) > 0,
            message=f"Added {len(synced_pages)} pages to sync list" + (f", {len(errors)} errors" if errors else ""),
            synced_pages=synced_pages,
            errors=errors if errors else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding pages to sync: {str(e)}")

@router.get("/sync/list")
async def list_sync_pages():
    """List all pages in the sync list."""
    try:
        pages = list(sync_pages_storage.values())
        return {
            "success": True,
            "pages": [page.dict() for page in pages],
            "count": len(pages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sync pages: {str(e)}")

class ConfluenceSyncRunRequest(BaseModel):
    """Request to run synchronization with credentials."""
    credentials: ConfluenceCredentials
    page_ids: Optional[List[str]] = None

@router.post("/sync/run")
async def run_sync(
    request: ConfluenceSyncRunRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Run synchronization for specified pages or all pages."""
    try:
        # Set up authentication headers from provided credentials
        auth_headers = get_auth_headers(request.credentials)
        
        # Determine which pages to sync
        pages_to_sync = []
        if request.page_ids:
            pages_to_sync = [sync_pages_storage[pid] for pid in request.page_ids if pid in sync_pages_storage]
        else:
            pages_to_sync = [page for page in sync_pages_storage.values() if page.sync_enabled]
        
        synced_count = 0
        errors = []
        
        for sync_page in pages_to_sync:
            try:
                # Create URL info for the sync page
                url_info = {
                    'page_id': sync_page.page_id,
                    'space_key': sync_page.space_key,
                    'api_base': sync_page.api_url.split('/rest/api/content/')[0],
                    'web_url': sync_page.web_url
                }
                
                # Fetch latest page content with proper auth
                page_info = await fetch_confluence_page_info(url_info, auth_headers)
                
                # Create document for the document service
                from llama_index.core import Document as LlamaDocument
                
                # Convert HTML content to plain text for better indexing
                content = page_info['content']
                if content:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')
                    plain_text = soup.get_text(separator='\n', strip=True)
                else:
                    plain_text = ""
                
                doc = LlamaDocument(
                    text=plain_text,
                    metadata={
                        'title': page_info['title'],
                        'source': 'confluence',
                        'page_id': page_info['page_id'],
                        'space_key': page_info['space_key'],
                        'web_url': sync_page.web_url,
                        'api_url': page_info['api_url'],
                        'sync_id': sync_page.id,
                        'document_type': 'confluence',
                        'created_at': page_info.get('created'),
                        'updated_at': page_info.get('updated')
                    }
                )
                
                # Process document through the main document service
                doc_id = f"confluence_sync_{sync_page.id}"
                
                # Add the document to the main service so it's available for chat
                await service.add_confluence_document(doc, doc_id)
                
                # Update sync timestamp
                sync_page.last_synced = datetime.now()
                sync_pages_storage[sync_page.id] = sync_page
                
                synced_count += 1
                
            except Exception as e:
                errors.append(f"Error syncing {sync_page.title}: {str(e)}")
        
        return {
            "success": synced_count > 0,
            "message": f"Synced {synced_count} pages" + (f", {len(errors)} errors" if errors else ""),
            "synced_count": synced_count,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running sync: {str(e)}")

@router.delete("/sync/{page_id}")
async def remove_from_sync(page_id: str):
    """Remove a page from the sync list."""
    try:
        if page_id in sync_pages_storage:
            removed_page = sync_pages_storage.pop(page_id)
            return {
                "success": True,
                "message": f"Removed '{removed_page.title}' from sync list"
            }
        else:
            raise HTTPException(status_code=404, detail="Page not found in sync list")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing page from sync: {str(e)}")

@router.post("/ingest/temporary", response_model=ConfluenceTemporaryIngestResponse)
async def ingest_page_temporarily(
    request: ConfluenceTemporaryIngestRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Temporarily ingest a Confluence page for chat conversations."""
    try:
        # Set up authentication headers
        auth_headers = get_auth_headers(request.credentials)
        
        # Parse URL and fetch page
        url_info = parse_confluence_url(request.web_url)
        page_info = await fetch_confluence_page_info(url_info, auth_headers)
        
        # Create temporary page ID
        temp_page_id = f"temp_confluence_{uuid.uuid4()}"
        expires_at = datetime.now() + timedelta(hours=2)  # Expire after 2 hours
        
        # Store temporarily (in production, use cache like Redis)
        temporary_pages_storage[temp_page_id] = {
            "page_id": temp_page_id,
            "original_page_id": url_info['page_id'],
            "title": page_info['title'],
            "content": page_info['content'],
            "space_key": page_info['space_key'],
            "web_url": request.web_url,
            "api_url": url_info['api_url'],
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        
        # Add to main document service for chat functionality
        try:
            from llama_index.core import Document as LlamaDocument
            
            # Convert HTML content to plain text
            content = page_info['content']
            if content:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                plain_text = soup.get_text(separator='\n', strip=True)
            else:
                plain_text = ""
            
            doc = LlamaDocument(
                text=plain_text,
                metadata={
                    'title': page_info['title'],
                    'source': 'confluence_temporary',
                    'page_id': url_info['page_id'],
                    'space_key': page_info['space_key'],
                    'web_url': request.web_url,
                    'api_url': url_info['api_url'],
                    'temp_page_id': temp_page_id,
                    'document_type': 'confluence_temporary',
                    'expires_at': expires_at.isoformat(),
                    'created_at': page_info.get('created'),
                    'updated_at': page_info.get('updated')
                }
            )
            
            # Add to document service with temporary ID
            await service.add_confluence_document(doc, temp_page_id)
            
        except Exception as e:
            print(f"Warning: Failed to add temporary page to document service: {e}")
        
        # Generate content preview (first 200 characters)
        content_preview = page_info['content'][:200] + "..." if len(page_info['content']) > 200 else page_info['content']
        
        return ConfluenceTemporaryIngestResponse(
            success=True,
            message=f"Temporarily ingested page '{page_info['title']}' - available for chat for 2 hours",
            page_id=temp_page_id,
            title=page_info['title'],
            content_preview=content_preview,
            expires_at=expires_at
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting page temporarily: {str(e)}")

@router.get("/ingest/temporary")
async def list_temporary_pages(service: DocumentService = Depends(get_document_service)):
    """List all temporarily ingested pages."""
    try:
        # Clean up expired pages
        current_time = datetime.now()
        expired_keys = [k for k, v in temporary_pages_storage.items() if v['expires_at'] < current_time]
        
        for key in expired_keys:
            del temporary_pages_storage[key]
            # Also remove from document service
            try:
                await service.delete_document(key)
                print(f"Cleaned up expired temporary page from document service: {key}")
            except Exception as e:
                print(f"Warning: Failed to clean up expired page from document service: {e}")
        
        # Return active pages
        active_pages = list(temporary_pages_storage.values())
        
        return {
            "success": True,
            "pages": active_pages,
            "count": len(active_pages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing temporary pages: {str(e)}")

@router.get("/ingest/temporary/{page_id}")
async def get_temporary_page(page_id: str):
    """Get content of a temporarily ingested page."""
    try:
        if page_id not in temporary_pages_storage:
            raise HTTPException(status_code=404, detail="Temporary page not found or expired")
        
        page_data = temporary_pages_storage[page_id]
        
        # Check if expired
        if page_data['expires_at'] < datetime.now():
            del temporary_pages_storage[page_id]
            raise HTTPException(status_code=404, detail="Temporary page has expired")
        
        return {
            "success": True,
            "page": page_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving temporary page: {str(e)}")

@router.delete("/ingest/temporary/{page_id}")
async def remove_temporary_page(
    page_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Remove a temporarily ingested page."""
    try:
        if page_id in temporary_pages_storage:
            removed_page = temporary_pages_storage.pop(page_id)
            
            # Also remove from document service
            try:
                await service.delete_document(page_id)
                print(f"Removed temporary page from document service: {page_id}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary page from document service: {e}")
            
            return {
                "success": True,
                "message": f"Removed temporary page '{removed_page['title']}'"
            }
        else:
            raise HTTPException(status_code=404, detail="Temporary page not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing temporary page: {str(e)}")

class ConfluencePageImportRequest(BaseModel):
    """Request to import a Confluence page as a permanent document."""
    credentials: ConfluenceCredentials
    web_url: str

class ConfluencePageImportResponse(BaseModel):
    """Response from importing a Confluence page as a document."""
    success: bool
    message: str
    title: Optional[str] = None
    page_id: Optional[str] = None

@router.post("/import-as-document", response_model=ConfluencePageImportResponse)
async def import_confluence_page_as_document(
    request: ConfluencePageImportRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Import a Confluence page as a permanent document."""
    try:
        # Set up authentication headers
        auth_headers = get_auth_headers(request.credentials)
        
        # Parse URL and fetch page
        url_info = parse_confluence_url(request.web_url)
        if not url_info:
            raise HTTPException(status_code=400, detail="Invalid Confluence URL format")
            
        page_info = await fetch_confluence_page_info(url_info, auth_headers)
        
        # Create permanent document ID using the resolved page_id
        doc_id = f"confluence_{page_info['page_id']}_{int(time.time())}"
        
        # Add to main document service as permanent document
        from llama_index.core import Document as LlamaDocument
        
        # Convert HTML content to plain text
        content = page_info['content']
        if content:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            plain_text = soup.get_text(separator='\n', strip=True)
        else:
            plain_text = ""
        
        doc = LlamaDocument(
            text=plain_text,
            metadata={
                'title': page_info['title'],
                'source': 'confluence_import',
                'page_id': page_info['page_id'],  # Use resolved page_id
                'space_key': page_info['space_key'],
                'web_url': request.web_url,
                'api_url': page_info['api_url'],  # Use resolved api_url
                'document_type': 'confluence',
                'created_at': page_info.get('created'),
                'updated_at': page_info.get('updated')
            }
        )
        
        # Add to document service as permanent document
        await service.add_confluence_document(doc, doc_id)
        
        return ConfluencePageImportResponse(
            success=True,
            message=f"Successfully imported '{page_info['title']}' as a document",
            title=page_info['title'],
            page_id=doc_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing Confluence page as document: {str(e)}") 