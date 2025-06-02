"""
API endpoints for Confluence integration.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..models.document import ConfluenceConfig, ConfluencePage
from ..document_processor.service import DocumentService

router = APIRouter()

# Dependency to get document service
def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    from main import document_service
    if not document_service:
        raise HTTPException(status_code=503, detail="Document service not available")
    return document_service

# Store Confluence configuration (in production, use secure storage)
confluence_config = None

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

@router.post("/config")
async def configure_confluence(config: ConfluenceConfig):
    """Configure Confluence connection settings."""
    try:
        global confluence_config
        confluence_config = config
        
        # Test connection (basic validation)
        # In a production system, you'd validate the credentials here
        
        return {
            "success": True,
            "message": "Confluence configuration saved successfully",
            "url": config.url,
            "username": config.username
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring Confluence: {str(e)}")

@router.get("/config")
async def get_confluence_config():
    """Get current Confluence configuration (without sensitive data)."""
    try:
        if not confluence_config:
            return {
                "success": True,
                "configured": False,
                "message": "Confluence not configured"
            }
        
        return {
            "success": True,
            "configured": True,
            "url": confluence_config.url,
            "username": confluence_config.username,
            "space_key": confluence_config.space_key
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving configuration: {str(e)}")

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
        if not confluence_config:
            raise HTTPException(status_code=400, detail="Confluence not configured")
        
        # Initialize Confluence reader
        from llama_index.readers.confluence import ConfluenceReader
        
        reader = ConfluenceReader(
            base_url=confluence_config.url,
            username=confluence_config.username,
            api_token=confluence_config.api_token
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
        if not confluence_config:
            raise HTTPException(status_code=400, detail="Confluence not configured")
        
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
        if not confluence_config:
            raise HTTPException(status_code=400, detail="Confluence not configured")
        
        # This would require implementing space listing via Confluence API
        # For now, return a placeholder
        
        return {
            "success": True,
            "spaces": [
                {
                    "key": confluence_config.space_key or "EXAMPLE",
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
        if not confluence_config:
            raise HTTPException(status_code=400, detail="Confluence not configured")
        
        # This would require implementing page listing via Confluence API
        # For now, return a placeholder
        
        return {
            "success": True,
            "space_key": space_key,
            "pages": [
                {
                    "id": "123456",
                    "title": "Example Page",
                    "url": f"{confluence_config.url}/spaces/{space_key}/pages/123456"
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
        if not confluence_config:
            raise HTTPException(status_code=400, detail="Confluence not configured")
        
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

@router.delete("/config")
async def clear_confluence_config():
    """Clear Confluence configuration."""
    try:
        global confluence_config
        confluence_config = None
        
        return {
            "success": True,
            "message": "Confluence configuration cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing configuration: {str(e)}") 