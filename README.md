# Document Assistant

An AI-powered application for analyzing documents and web pages, helping create new documents and Confluence pages offline using LlamaIndex, GPT4All, and React.

## ✨ Features

- **📄 Document Analysis**: Upload and analyze various document formats (PDF, DOCX, TXT, Markdown, HTML)
- **🤖 Dual AI Models**: 
  - **BERT Embeddings** - 6 advanced models for document understanding and semantic search
  - **GPT4All Language Models** - Local LLMs for chat and content generation (Llama 3, Mistral, etc.)
  - **Ollama Integration** - Advanced local LLM support with automatic process management
- **🔗 Confluence Integration**: Read existing pages and generate new Confluence content
- **💬 AI-Powered Chat**: Query your documents using natural language with intelligent context
  - **Background Processing** - Non-blocking chat with browser notifications
  - **Conversation History** - Configurable context management (0-50 messages)
  - **Smart Context Control** - Optimize token usage and response quality
- **✍️ Document Generation**: AI-assisted creation of new documents and pages
- **📊 Rich Metadata**: Advanced file analysis including content statistics, format-specific data
- **🛠️ Model Management**: Easy download, switch, and manage AI models through the UI
- **⚙️ Ollama Process Control**: Start, stop, restart, and monitor Ollama from the web interface
- **🚀 Auto-Start**: Optionally auto-start Ollama when the backend starts
- **🔒 Offline Operation**: Works completely offline - your data never leaves your device
- **🎨 Modern UI**: Beautiful React-based interface with tabbed settings and real-time status

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **4GB+ RAM** (for AI models)
- **10GB+ disk space** (for models and data)
- **Docker & Docker Compose** (optional)

### 1️⃣ Automated Setup

```bash
# Clone and setup
git clone <repository-url>
cd document-assistant
chmod +x setup.sh
./setup.sh
```

### 2️⃣ Download AI Models

The application uses three types of AI models:

**BERT Embedding Models** (downloaded automatically):
- `all-MiniLM-L6-v2` (384D) - Default, fast
- `all-mpnet-base-v2` (768D) - **Recommended** for best quality
- `bert-base-uncased`, `distilbert`, `roberta` - Additional options

**GPT4All Language Models** (download via UI or manually):

**Ollama Models** (managed through Ollama):
- Install [Ollama](https://ollama.ai) separately
- Download models via `ollama pull llama3.2:3b`
- Application provides automatic process management

**GPT4All Language Models** (download via UI or manually):

*Option A: Through the UI (Recommended)*
1. Start the application (step 3)
2. Go to Settings → Language Models
3. Click download on any recommended model

*Option B: Manual Download*
```bash
cd data/models

# Recommended models:
wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf  # 4.7GB
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf  # 2.2GB
```

### 3️⃣ Start the Application

**Local Development:**
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python main.py

# Terminal 2 - Frontend  
cd frontend
npm start
```

**Docker (Alternative):**
```bash
docker-compose up --build
```

### 4️⃣ Access the Application
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 🎯 Usage Guide

### Document Management
1. **Upload**: Drag & drop files on the Documents page
2. **Analyze**: View rich metadata including word counts, file properties, format-specific data
3. **Search**: Semantic search across all your documents

### AI Model Configuration
1. **Settings → Embedding Models**: Choose BERT model for document understanding
   - **MPNet**: Best quality for semantic search
   - **MiniLM**: Fastest for quick responses
   - **BERT Large**: Most accurate but slower

2. **Settings → Language Models**: Download and manage GPT4All models
   - **Llama 3**: Latest Meta model, excellent quality
   - **Phi-3 Mini**: Microsoft's efficient model  
   - **Mistral 7B**: Great for general tasks

3. **Settings → Ollama Management**: Control Ollama process and models
   - **Process Control**: Start, stop, restart Ollama from the UI
   - **Auto-Start**: Configure automatic startup when backend starts
   - **Model Management**: Switch between downloaded Ollama models
   - **Status Monitoring**: Real-time process and API status

### Chat & Generation
1. **Chat**: Ask questions about your documents
   - **Sync Mode**: Immediate responses (default)
   - **Background Mode**: Non-blocking processing with notifications
   - **Conversation Context**: AI remembers previous messages for continuity
2. **Generate**: Create new content based on your knowledge base
3. **Context**: AI automatically uses relevant document content

### Advanced Chat Settings
1. **Background Processing**: Enable non-blocking chat in Settings
   - Continue browsing while AI processes your request
   - Browser notifications when responses are ready
   - Responses appear in the correct conversation

2. **Conversation Context Management**: Control how much chat history is used
   - **Enable/Disable Context**: Toggle conversation memory on/off
   - **Max History**: Limit previous messages (0-50) for performance
   - **Token Optimization**: Reduce API costs and improve response times

### Confluence Integration  
1. **Configure**: Add Confluence credentials in Settings
2. **Import**: Sync existing pages for context
3. **Generate**: Create new Confluence content with AI assistance

## ⚙️ Configuration

### System Requirements by Model

| Model Type | Model | Size | RAM | Speed | Best For |
|------------|-------|------|-----|-------|----------|
| **Embedding** | MiniLM-L6-v2 | 23MB | 2GB | Fast | Quick search |
| **Embedding** | MPNet-base-v2 | 438MB | 4GB | Medium | **Recommended** |
| **Language** | Phi-3 Mini | 2.2GB | 6GB | Fast | General chat |
| **Language** | Llama 3 8B | 4.7GB | 8GB | Medium | **High quality** |
| **Language** | Mistral 7B | 4.1GB | 8GB | Medium | Balanced |
| **Ollama** | Llama 3.2:3b | 2.0GB | 4GB | Fast | **Recommended Ollama** |
| **Ollama** | Llama 3.2:1b | 1.3GB | 3GB | Very Fast | Lightweight |

### Environment Variables

Create `backend/.env`:
```env
# Basic Configuration
PORT=8000
HOST=0.0.0.0

# Model Paths (auto-configured)
GPT4ALL_MODEL_PATH=../data/models
EMBEDDING_MODEL=all-mpnet-base-v2

# Ollama (Optional)
OLLAMA_HOST=http://localhost:11434
AUTO_START_OLLAMA=false

# Confluence (Optional)
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USERNAME=your-email@domain.com  
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=YOUR_SPACE
```

### Application Settings

The `data/app_settings.json` file contains application configuration:
```json
{
  "embedding_model": "all-mpnet-base-v2",
  "llm_provider": "ollama",
  "preferred_ollama_model": "llama3.2:3b",
  "auto_start_ollama": false,
  "max_tokens": 512,
  "max_conversation_history": 10,
  "enable_conversation_context": true
}
```

**Key Settings**:
- `llm_provider`: "gpt4all" or "ollama"
- `auto_start_ollama`: Automatically start Ollama on backend startup
- `preferred_ollama_model`: Default Ollama model to use
- `max_conversation_history`: Maximum previous messages to include (0-50)
- `enable_conversation_context`: Whether to include chat history in responses

## Tech Stack

### Backend
- **LlamaIndex**: Document indexing and querying framework
- **GPT4All**: Local LLM inference for offline operation
- **FastAPI**: Modern, fast web framework for building APIs
- **ChromaDB**: Vector database for document embeddings
- **llama-index-readers-confluence**: Confluence integration

### Frontend
- **React 18**: Modern web application framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **React Router**: Client-side routing
- **Heroicons**: Beautiful SVG icons

## Project Structure

```
document-assistant/
├── backend/                 # Python backend API
│   ├── src/
│   │   ├── api/                # FastAPI route handlers
│   │   ├── document_processor/ # Document processing logic
│   │   ├── models/             # Pydantic data models
│   │   └── utils/              # Utility functions
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile             # Backend container config
│   └── main.py                # FastAPI application entry point
├── frontend/               # React web application
│   ├── src/
│   │   ├── components/         # Reusable React components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API service functions
│   │   └── utils/             # Utility functions
│   ├── package.json           # Node.js dependencies
│   ├── Dockerfile             # Frontend container config
│   └── public/               # Static assets
├── data/                   # Data storage (gitignored)
│   ├── documents/             # Uploaded documents
│   ├── confluence/            # Confluence cache
│   ├── models/               # GPT4All models (download separately)
│   └── chroma_db/            # Vector database
├── docker-compose.yml      # Container orchestration
├── setup.sh               # Automated setup script
└── README.md
```

## Port Configuration

| Service | Local Development | Docker | Description |
|---------|------------------|--------|-------------|
| Frontend | `http://localhost:3000` | `http://localhost:3000` | React web application |
| Backend API | `http://localhost:8000` | `http://localhost:8000` | FastAPI backend |
| API Docs | `http://localhost:8000/docs` | `http://localhost:8000/docs` | Interactive API documentation |
| ChromaDB | Local file storage | `http://localhost:8001` | Vector database (Docker only) |

## Important Notes

### File Size Considerations
- **GPT4All models are large** (1.9GB - 6.9GB each) and are **not included** in the repository
- **Models must be downloaded separately** before first use
- The `data/` directory is gitignored to prevent committing large files and user data

### What's Gitignored
- GPT4All model files (`*.gguf`, `*.bin`)
- User uploaded documents
- Vector database files
- Virtual environments
- Node.js dependencies
- Environment configuration files
- IDE/editor files
- OS-specific files

## Troubleshooting

### Common Issues

1. **"No GPT4All model found"**
   - Download a model file and place it in `./data/models/`
   - Ensure the filename matches the configuration

2. **Frontend build errors**
   - Delete `node_modules` and run `npm install` again
   - Ensure Node.js version is 18+

3. **Backend import errors**
   - Activate the virtual environment
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **Memory issues**
   - Use a smaller model (e.g., orca-mini-3b)
   - Increase system RAM or swap space

5. **Docker port conflicts**
   - Ensure ports 3000, 8000, and 8001 are available
   - Check if other services are using these ports
   - Modify docker-compose.yml port mappings if needed

6. **ChromaDB connection issues**
   - In Docker: ChromaDB runs on internal port 8000, exposed on 8001
   - In local development: Uses persistent file storage
   - Check logs: `docker-compose logs chroma`

### Performance Tips

- Use SSD storage for better model loading times
- Allocate sufficient RAM (8GB+ recommended)
- Close other applications when running large models
- Use smaller models for development/testing

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) for the document processing framework
- [GPT4All](https://github.com/nomic-ai/gpt4all) for offline LLM capabilities
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [FastAPI](https://github.com/tiangolo/fastapi) for the backend framework
- [React](https://github.com/facebook/react) for the frontend framework

## 🛠️ API Documentation

### Core Endpoints
- `POST /api/documents/upload` - Upload and process documents
- `GET /api/documents/` - List all documents with metadata
- `POST /api/chat/query` - Query documents with natural language
- `POST /api/chat/chat` - Chat with documents (with conversation history)
- `POST /api/chat/chat-background` - Start background chat processing
- `GET /api/chat/job/{job_id}` - Check background job status
- `DELETE /api/chat/job/{job_id}` - Clean up completed background job
- `POST /api/chat/generate` - Generate new content
- `POST /api/confluence/import` - Import Confluence pages

### Settings & Configuration API
- `GET /api/models/settings` - Get current application settings
- `POST /api/models/settings` - Update application settings (tokens, context, etc.)

### Model Management API  
- `GET /api/models/embeddings/available` - List BERT models
- `POST /api/models/embeddings/set` - Switch embedding model
- `GET /api/models/gpt4all/available` - List GPT4All models
- `POST /api/models/gpt4all/download` - Download language model
- `POST /api/models/gpt4all/upload` - Upload custom model

### Ollama Management API
- `GET /api/models/ollama/status` - Get Ollama process status
- `POST /api/models/ollama/start` - Start Ollama process
- `POST /api/models/ollama/stop` - Stop Ollama process  
- `POST /api/models/ollama/restart` - Restart Ollama process
- `GET /api/models/ollama/models` - List available Ollama models
- `POST /api/models/providers/set` - Switch between GPT4All and Ollama

**Full API Documentation**: http://localhost:8000/docs (when running)

## 🔧 Development

### Local Development
```bash
# Backend with auto-reload
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend with hot reload  
cd frontend
npm start
```

### Docker Development
```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Setup (Advanced)
```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install

# Create directories
mkdir -p data/{documents,confluence,models,chroma_db}
```

## 🚨 Troubleshooting

### Common Issues

**"No GPT4All model found"**
- Download a model via Settings → Language Models
- Or manually place `.gguf` files in `./data/models/`

**"Ollama model not available"**
- Install Ollama from https://ollama.ai
- Use Settings → Ollama Management to start Ollama
- Download models: `ollama pull llama3.2:3b`
- Switch provider in Settings → Language Models

**Memory/Performance Issues**  
- Use smaller models (Phi-3 Mini instead of Llama 3)
- Close other applications
- Ensure 8GB+ RAM for larger models

**Frontend Build Errors**
- Delete `node_modules`: `rm -rf node_modules && npm install`
- Ensure Node.js 18+

**Backend Import Errors**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**Docker Port Conflicts**
- Ensure ports 3000, 8000, 8001 are available
- Modify `docker-compose.yml` port mappings if needed

**Ollama Issues**
- **Installation**: Download from https://ollama.ai
- **Port conflicts**: Ollama uses port 11434 by default
- **Process not starting**: Check Settings → Ollama Management for status
- **Models not loading**: Download with `ollama pull model-name`
- **API not responding**: Restart via Settings or `ollama serve`
- **Auto-start failed**: Check logs and manually start Ollama first

### Device Support
- **NVIDIA GPU**: CUDA support automatically detected
- **Apple Silicon**: MPS acceleration enabled  
- **CPU Only**: Automatic fallback, works on any system

## 📂 Project Structure

```
document-assistant/
├── backend/                    # Python FastAPI backend
│   ├── src/
│   │   ├── api/               # REST API endpoints
│   │   ├── document_processor/ # Core processing logic
│   │   ├── models/            # Data models
│   │   └── utils/             # Utilities
│   ├── requirements.txt       # Python dependencies
│   └── main.py               # Application entry
├── frontend/                  # React TypeScript frontend
│   ├── src/
│   │   ├── components/        # Reusable components
│   │   ├── pages/            # Page components  
│   │   ├── services/         # API services
│   │   └── utils/            # Utilities
│   └── package.json          # Node.js dependencies
├── data/                     # Data storage (gitignored)
│   ├── documents/           # Uploaded files
│   ├── models/             # AI model files
│   └── chroma_db/          # Vector database
├── docker-compose.yml       # Container orchestration
├── setup.sh                # Automated setup
└── README.md               # This file
```

## Important Notes

### What's Included vs. Downloaded
- ✅ **Application code** - Ready to run
- ✅ **BERT models** - Downloaded automatically  
- ❌ **GPT4All models** - Download separately (1-5GB each)
- ❌ **User data** - Your documents stay private

### Privacy & Security
- **100% Offline** - No data sent to external servers
- **Local Processing** - All AI runs on your device
- **Private Storage** - Documents and models stay on your machine
- **No Tracking** - No analytics or usage monitoring

### Performance Guidelines
- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB RAM, 10GB disk space  
- **Optimal**: 16GB RAM, 20GB disk space, dedicated GPU

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with clear description

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) - Document processing framework
- [GPT4All](https://github.com/nomic-ai/gpt4all) - Offline LLM capabilities  
- [HuggingFace](https://huggingface.co/) - BERT models and transformers
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector storage
- [FastAPI](https://github.com/tiangolo/fastapi) - Backend framework
- [React](https://github.com/facebook/react) - Frontend framework 