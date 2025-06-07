# Developer Documentation

This document contains technical implementation details for contributors and developers working on the Document Assistant codebase.

## 🏗️ Architecture Overview

### Backend Architecture
- **FastAPI** - REST API framework
- **LlamaIndex** - Document processing and indexing
- **ChromaDB** - Vector database for embeddings
- **Async/Await** - Non-blocking operations
- **Pydantic** - Data validation and serialization

### Frontend Architecture  
- **React 18** - Component-based UI
- **TypeScript** - Type safety
- **Material-UI** - Component library
- **React Router** - Client-side routing
- **Service Layer** - API abstractions

## 🤖 AI Model Integration

### Ollama Integration

**Implementation**: `backend/src/api/models.py` (lines 1047-1354) and `backend/src/document_processor/service.py` (lines 307-430)

**Process Management**:
- **Auto-detection**: Platform-specific process detection (macOS, Linux, Windows)
- **Multiple start methods**: brew services, systemctl, direct execution
- **Status monitoring**: Real-time process and API status checking
- **Control capability**: Determines if Ollama can be controlled based on command availability

**Key Features**:
```python
# Ollama Process Status
{
    "running": bool,        # Process is running
    "responding": bool,     # API is responding
    "process_id": str,      # Process ID if running
    "can_control": bool,    # Can start/stop/restart
    "version": str,         # Ollama version
    "models": List[str],    # Available models
    "platform": str         # Operating system
}
```

**Auto-start Configuration**: 
- Configured via `data/app_settings.json`
- Triggered on backend startup if enabled
- Supports platform-specific startup methods

### BERT Embedding Models

**Implementation**: `backend/src/document_processor/service.py` (lines 33-174)

**Supported Models**:
```python
AVAILABLE_MODELS = {
    "sentence-bert": {
        "model_name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "Fast and efficient for sentence embeddings"
    },
    "mpnet": {
        "model_name": "all-mpnet-base-v2", 
        "dimensions": 768,
        "description": "Best quality for semantic search (recommended)"
    },
    "bert-base-uncased": {
        "model_name": "bert-base-uncased",
        "dimensions": 768,
        "description": "Standard BERT model"
    },
    "bert-large": {
        "model_name": "bert-large-uncased",
        "dimensions": 1024,
        "description": "Larger BERT with better accuracy"
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "dimensions": 768,
        "description": "Faster, smaller BERT variant"
    },
    "roberta": {
        "model_name": "sentence-transformers/all-roberta-large-v1",
        "dimensions": 1024,
        "description": "RoBERTa-based sentence transformer"
    }
}
```

**Device Selection Logic**:
1. Check if GPU explicitly disabled
2. Prefer CUDA if available (NVIDIA GPUs)
3. Use MPS on Apple Silicon (M1/M2 Macs)
4. Fallback to CPU for all systems

**Loading Strategy**:
1. Try SentenceTransformer (optimized for embeddings)
2. Fallback to direct transformers if SentenceTransformer fails
3. Error handling with zero-vector fallback

### GPT4All Language Models

**Implementation**: `backend/src/api/models.py`

**Pre-configured Models**:
```python
AVAILABLE_GPT4ALL_MODELS = {
    "llama-3-8b-instruct": {
        "filename": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "name": "Llama 3 8B Instruct",
        "size_bytes": 4661000000,
        "description": "Meta's latest model - excellent for instruction following",
        "download_url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "recommended": True
    },
    # ... more models
}
```

## 📊 Document Metadata System

**Implementation**: `backend/src/document_processor/service.py`

### File Metadata Extraction
- **File system properties**: size, timestamps, permissions
- **MIME type detection**: content type identification
- **File hashing**: SHA-256 checksums for deduplication
- **Platform info**: OS and Python version tracking

### Content Metadata Analysis
- **Text statistics**: word/line/paragraph counts
- **Character analysis**: encoding and special characters
- **Language detection**: automatic language identification

### Format-Specific Extractors

**PDF Files**:
```python
def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
    # Extract pages, author, title, creation/modification dates
    # Handles encrypted PDFs and metadata parsing errors
```

**DOCX Files**:
```python
def _extract_docx_metadata(self, file_path: Path) -> Dict[str, Any]:
    # Extract paragraph count and core document properties
    # Handles corrupted files gracefully
```

**Markdown Files**:
```python  
def _extract_markdown_metadata(self, content: str) -> Dict[str, Any]:
    # Count headers by level, extract links and code blocks
    # Parse front matter if present
```

## 🔌 API Endpoints

### Model Management

**BERT Embedding Models**:
- `GET /api/models/embeddings/available` - List available models
- `GET /api/models/embeddings/current` - Get current model info
- `POST /api/models/embeddings/set` - Change embedding model
- `POST /api/models/embeddings/test` - Test model with sample text

**GPT4All Language Models**:
- `GET /api/models/gpt4all/available` - List available models
- `GET /api/models/gpt4all/downloaded` - List downloaded models
- `POST /api/models/gpt4all/download` - Download model from URL
- `POST /api/models/gpt4all/upload` - Upload custom model file
- `DELETE /api/models/gpt4all/{filename}` - Delete model
- `POST /api/models/gpt4all/set-active` - Set active model

**Ollama Integration**:
- `GET /api/models/ollama/status` - Get detailed process status
- `POST /api/models/ollama/start` - Start Ollama process
- `POST /api/models/ollama/stop` - Stop Ollama process
- `POST /api/models/ollama/restart` - Restart Ollama process
- `GET /api/models/ollama/models` - List available Ollama models
- `GET /api/models/providers/current` - Get current LLM provider
- `POST /api/models/providers/set` - Switch between GPT4All and Ollama

### Document Operations
- `POST /api/documents/upload` - Upload with metadata extraction
- `GET /api/documents/{id}/metadata` - Get detailed metadata
- `GET /api/documents/metadata/all` - Bulk metadata retrieval

## 🎨 Frontend Components

### Settings Page Architecture

**Tab Structure**:
```typescript
// Organized into logical sections
enum SettingsTabs {
  EMBEDDING_MODELS = 0,
  LANGUAGE_MODELS = 1,
  STORAGE_SYSTEM = 2,
  APPLICATION = 3
}
```

**Ollama Management Section**:
```typescript
// State management for Ollama process control
const [ollamaStatus, setOllamaStatus] = useState<any>(null);
const [loadingOllamaStatus, setLoadingOllamaStatus] = useState(false);
const [controllingOllama, setControllingOllama] = useState(false);
const [autoStartOllama, setAutoStartOllama] = useState(false);

// Process control functions
const handleStartOllama = async () => { /* ... */ };
const handleStopOllama = async () => { /* ... */ };
const handleRestartOllama = async () => { /* ... */ };
```

**State Management**:
```typescript
// Model state
const [gpt4allModels, setGpt4allModels] = useState<any[]>([]);
const [downloadedModels, setDownloadedModels] = useState<any[]>([]);
const [downloadingModels, setDownloadingModels] = useState<Set<string>>();
const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>();

// UI state
const [currentTab, setCurrentTab] = useState(0);
const [uploadingModel, setUploadingModel] = useState(false);
```

**Progress Polling**:
```typescript
const pollDownloadProgress = useCallback((filename: string) => {
  const interval = setInterval(async () => {
    const status = await modelsAPI.getDownloadStatus(filename);
    if (status.is_complete) {
      clearInterval(interval);
      // Update UI and refresh model list
    }
  }, 2000);
}, []);
```

## 🧪 Testing Strategy

### Backend Testing
```bash
# Run with virtual environment activated
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

### Frontend Testing  
```bash
cd frontend
npm test
```

### Manual Testing Checklist
- [ ] Document upload and processing
- [ ] Model switching and validation
- [ ] Download progress tracking
- [ ] Error handling and recovery
- [ ] Cross-platform compatibility

## 🚀 Deployment Considerations

### Docker Multi-Stage Builds
- **Development**: Hot reload, source mounting
- **Production**: Optimized builds, minimal images

### Environment Variables
```env
# Development
NODE_ENV=development
REACT_APP_API_URL=http://localhost:8000

# Production  
NODE_ENV=production
REACT_APP_API_URL=https://your-domain.com/api
```

### Performance Optimization
- **Model Caching**: Persistent model storage
- **Lazy Loading**: Models loaded on-demand
- **Memory Management**: Automatic cleanup for large models
- **Async Operations**: Non-blocking UI updates

## 🔧 Development Workflow

### Code Style
```bash
# Backend formatting
black backend/src/
isort backend/src/

# Frontend formatting
cd frontend
npm run format
npm run lint
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Adding New Models

**BERT Models**:
1. Add to `AVAILABLE_MODELS` in `service.py`
2. Test model loading and embedding generation
3. Update frontend model dropdown

**GPT4All Models**:
1. Add to `AVAILABLE_GPT4ALL_MODELS` in `models.py`  
2. Verify download URL and file integrity
3. Test model activation and inference

## 📈 Monitoring & Debugging

### Logging
```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Use throughout code
logger.info("Model loaded successfully")
logger.error("Failed to process document", exc_info=True)
```

### Performance Profiling
```python
import cProfile
import pstats

# Profile model loading
pr = cProfile.Profile()
pr.enable()
# ... model operations
pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats('cumulative').print_stats(10)
```

### Frontend Debugging
```typescript
// React DevTools
// Redux DevTools (if using Redux)
// Browser performance profiling

// Async debugging
console.time('model-switch');
await modelsAPI.setEmbeddingModel(modelKey);
console.timeEnd('model-switch');
```

## 🤝 Contributing Guidelines

### Branch Naming
- `feature/model-management` - New features
- `bugfix/download-progress` - Bug fixes  
- `refactor/api-cleanup` - Code improvements
- `docs/api-updates` - Documentation changes

### Pull Request Process
1. Create feature branch from main
2. Implement changes with tests
3. Update documentation if needed
4. Submit PR with clear description
5. Address review feedback
6. Merge after approval

### Code Review Checklist
- [ ] Type safety (TypeScript/Pydantic)
- [ ] Error handling and edge cases
- [ ] Performance implications
- [ ] Security considerations
- [ ] Documentation updates
- [ ] Test coverage 