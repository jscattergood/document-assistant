"""
Web import API for importing accessible web pages.
"""
import time
import requests
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
# Try to import readability, fallback gracefully if not available
try:
    from readability import Document as ReadabilityDocument
    READABILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: readability library not available: {e}")
    ReadabilityDocument = None
    READABILITY_AVAILABLE = False

from ..document_processor.service import DocumentService

router = APIRouter(prefix="/web-import", tags=["web-import"])

def get_document_service() -> DocumentService:
    """Get document service instance."""
    from ..document_processor.service import DocumentService
    return DocumentService()

class WebPageImportRequest(BaseModel):
    """Request to import a web page as a permanent document."""
    url: str = Field(..., description="URL of the web page to import")
    extract_mode: str = Field(default="auto", description="Content extraction mode: 'auto', 'readability', or 'full'")

class WebPageImportResponse(BaseModel):
    """Response from importing a web page as a document."""
    success: bool
    message: str
    title: Optional[str] = None
    url: str
    document_id: Optional[str] = None

def is_valid_url(url: str) -> bool:
    """Validate if the URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def extract_content_auto(html_content: str, url: str) -> dict:
    """Automatically extract the best content from HTML using multiple methods."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try readability first for better content extraction (if available)
    if READABILITY_AVAILABLE:
        try:
            doc = ReadabilityDocument(html_content)
            readability_title = doc.title()
            readability_content = doc.summary()
            
            if readability_content:
                # Convert readability HTML to plain text
                content_soup = BeautifulSoup(readability_content, 'html.parser')
                plain_text = content_soup.get_text(separator='\n', strip=True)
                
                return {
                    'title': readability_title or soup.title.string if soup.title else 'Untitled',
                    'content': plain_text,
                    'method': 'readability'
                }
        except Exception as e:
            print(f"Readability extraction failed: {e}")
    
    # Fallback to manual extraction
    return extract_content_manual(soup, url)

def extract_content_manual(soup: BeautifulSoup, url: str) -> dict:
    """Manually extract content using common HTML patterns."""
    # Extract title
    title = None
    if soup.title:
        title = soup.title.string.strip()
    
    # Try common title selectors
    if not title:
        for selector in ['h1', '.title', '#title', '.post-title', '.article-title']:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                break
    
    title = title or 'Untitled'
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
        element.decompose()
    
    # Try to find main content area
    content_element = None
    
    # Common content selectors (ordered by specificity)
    content_selectors = [
        'article',
        '.content',
        '.post-content',
        '.article-content',
        '.entry-content',
        'main',
        '.main-content',
        '#content',
        '.container .row',
        'body'
    ]
    
    for selector in content_selectors:
        element = soup.select_one(selector)
        if element:
            content_element = element
            break
    
    if not content_element:
        content_element = soup.find('body') or soup
    
    # Extract text content
    content = content_element.get_text(separator='\n', strip=True)
    
    # Clean up the content
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:  # Filter out very short lines
            cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    return {
        'title': title,
        'content': content,
        'method': 'manual'
    }

def extract_content_full(soup: BeautifulSoup, url: str) -> dict:
    """Extract full page content with minimal filtering."""
    title = soup.title.string.strip() if soup.title else 'Untitled'
    
    # Remove only script and style tags
    for element in soup(['script', 'style']):
        element.decompose()
    
    content = soup.get_text(separator='\n', strip=True)
    
    return {
        'title': title,
        'content': content,
        'method': 'full'
    }

async def fetch_web_page(url: str) -> dict:
    """Fetch and parse a web page."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        # Provide specific error messages for common HTTP status codes
        if response.status_code == 404:
            raise HTTPException(
                status_code=400, 
                detail=f"Page not found (404). Please check the URL and make sure the page exists: {url}"
            )
        elif response.status_code == 403:
            raise HTTPException(
                status_code=400,
                detail="Access forbidden (403). The website doesn't allow automated access to this page."
            )
        elif response.status_code == 429:
            raise HTTPException(
                status_code=400,
                detail="Too many requests (429). The website is rate-limiting our requests. Please try again later or use a different URL."
            )
        elif response.status_code == 500:
            raise HTTPException(
                status_code=400,
                detail="Server error (500). The website is experiencing technical difficulties. Please try again later."
            )
        elif response.status_code >= 400:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to access webpage (HTTP {response.status_code}). Please check the URL or try a different page."
            )
        
        response.raise_for_status()  # This will raise for any remaining HTTP errors
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            if 'text/plain' in content_type:
                raise HTTPException(
                    status_code=400, 
                    detail="This URL points to a plain text file, not a webpage. Please use a URL that points to an HTML page."
                )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"This URL doesn't point to a webpage (content-type: {content_type}). Please use a URL that points to an HTML page."
                )
        
        return {
            'html': response.text,
            'final_url': response.url,
            'status_code': response.status_code
        }
        
    except HTTPException:
        raise  # Re-raise our custom HTTP exceptions
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=400,
            detail="Request timed out. The webpage took too long to respond. Please try a different URL."
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=400,
            detail="Connection failed. Please check the URL and your internet connection."
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch webpage: {str(e)}"
        )

@router.post("/import-page", response_model=WebPageImportResponse)
async def import_web_page(
    request: WebPageImportRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Import a web page as a permanent document."""
    try:
        # Validate URL
        if not is_valid_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Fetch the web page
        page_data = await fetch_web_page(request.url)
        html_content = page_data['html']
        final_url = page_data['final_url']
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract content based on mode
        if request.extract_mode == "readability":
            if READABILITY_AVAILABLE:
                try:
                    doc = ReadabilityDocument(html_content)
                    title = doc.title()
                    content_html = doc.summary()
                    content_soup = BeautifulSoup(content_html, 'html.parser')
                    content = content_soup.get_text(separator='\n', strip=True)
                    extraction_method = 'readability'
                except Exception:
                    # Fallback to manual if readability fails
                    result = extract_content_manual(soup, final_url)
                    title = result['title']
                    content = result['content']
                    extraction_method = 'manual_fallback'
            else:
                # Readability not available, use manual extraction
                result = extract_content_manual(soup, final_url)
                title = result['title']
                content = result['content']
                extraction_method = 'manual_no_readability'
        elif request.extract_mode == "full":
            result = extract_content_full(soup, final_url)
            title = result['title']
            content = result['content']
            extraction_method = 'full'
        else:  # auto
            result = extract_content_auto(html_content, final_url)
            title = result['title']
            content = result['content']
            extraction_method = result['method']
        
        if not content or len(content.strip()) < 100:
            raise HTTPException(status_code=400, detail="Could not extract meaningful content from the web page")
        
        # Create document ID
        doc_id = f"web_{int(time.time())}_{hash(final_url) % 10000}"
        
        # Add to document service
        from llama_index.core import Document as LlamaDocument
        
        # Extract domain for metadata
        parsed_url = urlparse(final_url)
        domain = parsed_url.netloc
        
        doc = LlamaDocument(
            text=content,
            metadata={
                'title': title,
                'source': 'web_import',
                'url': final_url,
                'original_url': request.url,
                'domain': domain,
                'document_type': 'webpage',
                'extraction_method': extraction_method,
                'imported_at': time.time()
            }
        )
        
        # Add to document service
        await service.add_web_document(doc, doc_id)
        
        return WebPageImportResponse(
            success=True,
            message=f"Successfully imported webpage: {title}",
            title=title,
            url=final_url,
            document_id=doc_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing web page: {str(e)}")

@router.post("/preview-url")
async def preview_web_url(request: dict):
    """Preview a URL to see if it's accessible and get basic info before importing."""
    try:
        url = request.get('url')
        if not url:
            return {
                'success': False,
                'accessible': False,
                'error': "URL is required"
            }
        
        if not is_valid_url(url):
            return {
                'success': False,
                'accessible': False,
                'error': "Invalid URL format. Please include http:// or https://"
            }
        
        # Fetch page info
        page_data = await fetch_web_page(url)
        soup = BeautifulSoup(page_data['html'], 'html.parser')
        
        title = soup.title.string.strip() if soup.title else 'No title found'
        
        # Get basic page info
        description = None
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content', '')
        
        return {
            'success': True,
            'accessible': True,
            'title': title,
            'description': description,
            'final_url': page_data['final_url'],
            'content_type': 'text/html'
        }
        
    except HTTPException as http_ex:
        # Convert HTTP errors (404, 403, 429, etc.) to JSON responses
        return {
            'success': False,
            'accessible': False,
            'error': http_ex.detail
        }
    except Exception as e:
        return {
            'success': False,
            'accessible': False,
            'error': f"Unexpected error: {str(e)}"
        } 