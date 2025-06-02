"""
Document processing service using LlamaIndex and GPT4All.
"""
import os
import uuid
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    Settings,
    Document as LlamaDocument
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import MockLLM, LLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.base.llms.types import LLMMetadata, CompletionResponse, ChatResponse, ChatMessage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.confluence import ConfluenceReader
import chromadb
from pypdf import PdfReader
from docx import Document as DocxDocument
import markdown
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from ..models.document import Document, DocumentType, DocumentStatus, ConfluencePage

class HuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace embedding model wrapper."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

class GPT4AllLLM(LLM):
    """GPT4All LLM wrapper that properly inherits from LlamaIndex LLM."""
    
    model_path: str
    model_name: str
    
    def __init__(self, model_path: str, model_name: str, **kwargs):
        super().__init__(model_path=model_path, model_name=model_name, **kwargs)
        self._model = None
        self._available = False
        self._init_model()
    
    def _init_model(self):
        try:
            from gpt4all import GPT4All
            self._model = GPT4All(self.model_name, model_path=self.model_path)
            self._available = True
            print(f"Successfully loaded GPT4All model: {self.model_name}")
        except Exception as e:
            print(f"Failed to load GPT4All model: {e}")
            self._available = False
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            num_output=512,
            model_name=self.model_name,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        if not self._available or self._model is None:
            return CompletionResponse(text="GPT4All model not available. Please check model installation.")
        
        try:
            max_tokens = kwargs.get('max_tokens', 512)
            response = self._model.generate(prompt, max_tokens=max_tokens)
            return CompletionResponse(text=response)
        except Exception as e:
            return CompletionResponse(text=f"Error generating response: {e}")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        # For simplicity, just return the complete response
        response = self.complete(prompt, **kwargs)
        yield response
    
    @llm_completion_callback()
    def chat(self, messages, **kwargs):
        # Convert chat messages to a single prompt
        prompt = ""
        for message in messages:
            role = message.role
            content = message.content
            prompt += f"{role}: {content}\n"
        
        prompt += "assistant: "
        response = self.complete(prompt, **kwargs)
        
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response.text)
        )
    
    @llm_completion_callback()
    def stream_chat(self, messages, **kwargs):
        # For simplicity, just return the complete chat response
        response = self.chat(messages, **kwargs)
        yield response
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        # Async version - for simplicity, just call the sync version
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs):
        # Async streaming version
        response = await self.acomplete(prompt, **kwargs)
        yield response
    
    async def achat(self, messages, **kwargs):
        # Async chat version
        return self.chat(messages, **kwargs)
    
    async def astream_chat(self, messages, **kwargs):
        # Async streaming chat version
        response = await self.achat(messages, **kwargs)
        yield response

class DocumentService:
    """Service for processing and managing documents with LlamaIndex."""
    
    def __init__(self):
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.documents: Dict[str, Document] = {}
        self.chroma_client = None
        self.vector_store = None
        self.llm = None
        
        # Paths
        self.data_dir = Path("../data")
        self.documents_dir = self.data_dir / "documents"
        self.models_dir = self.data_dir / "models"
        self.chroma_dir = self.data_dir / "chroma_db"
        
        # Ensure directories exist
        for directory in [self.data_dir, self.documents_dir, self.models_dir, self.chroma_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize the document service with LLM and embeddings."""
        try:
            # Initialize GPT4All LLM
            model_path = self._get_gpt4all_model_path()
            if model_path:
                self.llm = GPT4AllLLM(str(self.models_dir), model_path)
                Settings.llm = self.llm
                print(f"Initialized GPT4All with model: {model_path}")
            else:
                print("Warning: No GPT4All model found. Using mock LLM.")
                Settings.llm = MockLLM()
            
            # Configure embeddings and node parser
            Settings.embed_model = HuggingFaceEmbedding("all-MiniLM-L6-v2")
            Settings.node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            
            # Initialize Chroma vector store
            # Check if running in Docker environment
            chroma_host = os.getenv('CHROMA_HOST')
            chroma_port = int(os.getenv('CHROMA_PORT', '8000'))
            
            if chroma_host:
                # Running in Docker - connect to ChromaDB service
                print(f"Connecting to ChromaDB at {chroma_host}:{chroma_port}")
                self.chroma_client = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=chromadb.Settings(
                        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                        chroma_client_auth_credentials="test-token"
                    )
                )
            else:
                # Running locally - use persistent client
                print(f"Using local ChromaDB at {self.chroma_dir}")
                self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            
            collection = self.chroma_client.get_or_create_collection("documents")
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Initialize or load existing index
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context
                )
                print("Loaded existing document index")
            except Exception:
                # Create new index if none exists
                self.index = VectorStoreIndex([], storage_context=storage_context)
                print("Created new document index")
            
            # Initialize query and chat engines
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=3,
                response_mode="tree_summarize"
            )
            
            self.chat_engine = self.index.as_chat_engine(
                chat_mode="condense_question",
                verbose=True
            )
            
            print("Document service initialized successfully")
            
        except Exception as e:
            print(f"Error initializing document service: {e}")
            raise
    
    def _get_gpt4all_model_path(self) -> Optional[str]:
        """Get the path to an available GPT4All model."""
        model_extensions = [".gguf", ".bin"]
        
        for model_file in self.models_dir.iterdir():
            if model_file.is_file() and any(model_file.suffix == ext for ext in model_extensions):
                return model_file.name
        
        return None
    
    async def process_uploaded_file(self, file_path: str, filename: str) -> Document:
        """Process an uploaded file and add it to the index."""
        try:
            # Determine document type
            doc_type = self._get_document_type(filename)
            
            # Extract content based on file type
            content = await self._extract_content(file_path, doc_type)
            
            # Create document record
            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id,
                title=filename,
                type=doc_type,
                content=content,
                file_path=file_path,
                status=DocumentStatus.PROCESSING,
                size_bytes=os.path.getsize(file_path)
            )
            
            # Add to documents dictionary
            self.documents[doc_id] = document
            
            # Create LlamaIndex document and add to index
            llama_doc = LlamaDocument(
                text=content,
                metadata={
                    "doc_id": doc_id,
                    "title": filename,
                    "type": doc_type.value,
                    "file_path": file_path
                }
            )
            
            # Add to index
            self.index.insert(llama_doc)
            
            # Update status
            document.status = DocumentStatus.INDEXED
            self.documents[doc_id] = document
            
            print(f"Successfully processed document: {filename}")
            return document
            
        except Exception as e:
            print(f"Error processing document {filename}: {e}")
            if doc_id in self.documents:
                self.documents[doc_id].status = DocumentStatus.ERROR
            raise
    
    def _get_document_type(self, filename: str) -> DocumentType:
        """Determine document type from filename."""
        extension = Path(filename).suffix.lower()
        
        type_mapping = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".txt": DocumentType.TXT,
            ".md": DocumentType.MARKDOWN,
            ".markdown": DocumentType.MARKDOWN,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML
        }
        
        return type_mapping.get(extension, DocumentType.TXT)
    
    async def _extract_content(self, file_path: str, doc_type: DocumentType) -> str:
        """Extract text content from different file types."""
        try:
            if doc_type == DocumentType.PDF:
                return await self._extract_pdf_content(file_path)
            elif doc_type == DocumentType.DOCX:
                return await self._extract_docx_content(file_path)
            elif doc_type == DocumentType.TXT:
                return await self._extract_text_content(file_path)
            elif doc_type == DocumentType.MARKDOWN:
                return await self._extract_markdown_content(file_path)
            elif doc_type == DocumentType.HTML:
                return await self._extract_html_content(file_path)
            else:
                return await self._extract_text_content(file_path)
                
        except Exception as e:
            print(f"Error extracting content from {file_path}: {e}")
            return ""
    
    async def _extract_pdf_content(self, file_path: str) -> str:
        """Extract text from PDF file."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    
    async def _extract_docx_content(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    async def _extract_text_content(self, file_path: str) -> str:
        """Extract text from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    async def _extract_markdown_content(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    
    async def _extract_html_content(self, file_path: str) -> str:
        """Extract text from HTML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
    
    async def query_documents(self, query: str, document_ids: Optional[List[str]] = None) -> str:
        """Query the document index with a natural language question."""
        try:
            if not self.query_engine:
                raise ValueError("Query engine not initialized")
            
            # If specific documents are requested, we could filter here
            # For now, query across all documents
            response = self.query_engine.query(query)
            return str(response)
            
        except Exception as e:
            print(f"Error querying documents: {e}")
            return f"Sorry, I encountered an error while processing your query: {e}"
    
    async def chat_with_documents(self, message: str, conversation_history: Optional[List] = None) -> str:
        """Chat with documents using the chat engine."""
        try:
            if not self.chat_engine:
                raise ValueError("Chat engine not initialized")
            
            # Reset chat engine if new conversation
            if not conversation_history:
                self.chat_engine.reset()
            
            response = self.chat_engine.chat(message)
            return str(response)
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return f"Sorry, I encountered an error: {e}"
    
    def get_all_documents(self) -> List[Document]:
        """Get all processed documents."""
        return list(self.documents.values())
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a specific document by ID."""
        return self.documents.get(doc_id)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index and storage."""
        try:
            if doc_id not in self.documents:
                return False
            
            document = self.documents[doc_id]
            
            # Remove file if it exists
            if document.file_path and os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            # Remove from documents dictionary
            del self.documents[doc_id]
            
            # Note: ChromaDB doesn't have a direct way to remove by metadata
            # In a production system, you'd want to track document IDs in the vector store
            
            return True
            
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False
    
    async def add_confluence_document(self, llama_doc: LlamaDocument, doc_id: str) -> Document:
        """Add a Confluence document to the main document service."""
        try:
            # Create a Document object for tracking
            document = Document(
                id=doc_id,
                title=llama_doc.metadata.get('title', 'Untitled Confluence Page'),
                type=DocumentType.CONFLUENCE,
                content=llama_doc.text,
                status=DocumentStatus.PROCESSING,
                metadata=llama_doc.metadata,
                file_path=None,  # Confluence pages don't have files
                size_bytes=len(llama_doc.text.encode('utf-8')) if llama_doc.text else 0
            )
            
            # Add to the vector index
            if self.index is not None:
                # Insert the document into the index
                self.index.insert(llama_doc)
                
                # Update query and chat engines
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=3,
                    response_mode="tree_summarize"
                )
                
                self.chat_engine = self.index.as_chat_engine(
                    chat_mode="condense_question",
                    verbose=True
                )
            
            # Mark as indexed
            document.status = DocumentStatus.INDEXED
            
            # Store in documents dictionary
            self.documents[doc_id] = document
            
            print(f"Successfully added Confluence document: {document.title}")
            return document
            
        except Exception as e:
            print(f"Error adding Confluence document {doc_id}: {e}")
            # Create error document
            document = Document(
                id=doc_id,
                title=llama_doc.metadata.get('title', 'Untitled Confluence Page'),
                type=DocumentType.CONFLUENCE,
                status=DocumentStatus.ERROR,
                metadata=llama_doc.metadata
            )
            self.documents[doc_id] = document
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.chroma_client:
            # ChromaDB client cleanup
            pass
        print("Document service cleanup complete") 