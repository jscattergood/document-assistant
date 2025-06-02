# Document Assistant

An AI-powered application for analyzing documents and web pages, helping create new documents and Confluence pages offline using LlamaIndex, GPT4All, and React.

## Features

- **📄 Document Analysis**: Upload and analyze various document formats (PDF, DOCX, TXT, Markdown, HTML)
- **🔗 Confluence Integration**: Read existing pages and generate new Confluence content
- **💬 AI-Powered Chat**: Query your documents using natural language
- **✍️ Document Generation**: AI-assisted creation of new documents and pages
- **🔒 Offline Operation**: Works completely offline using local LLM inference with GPT4All
- **🎨 Modern UI**: Beautiful React-based web interface with Tailwind CSS

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

## Quick Start

### Prerequisites
- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **4GB+ RAM** (for GPT4All models)
- **10GB+ disk space** (for models and data)
- **Docker & Docker Compose** (optional, for containerized deployment)

### Automated Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd document-assistant
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Download GPT4All models** (required for AI functionality):
   
   **Option A: Automatic download (recommended)**
   ```bash
   # Download the smaller, faster model (1.9GB)
   cd data/models
   wget https://gpt4all.io/models/gguf/orca-mini-3b-gguf2-q4_0.gguf
   
   # Or download the larger, more capable model (6.9GB)
   wget https://gpt4all.io/models/gguf/nous-hermes-llama2-13b.Q4_0.gguf
   ```
   
   **Option B: Manual download**
   - Visit [GPT4All Downloads](https://gpt4all.io/index.html)
   - Download a `.gguf` model file
   - Place it in `./data/models/`

### Deployment Options

#### Option 1: Local Development

4. **Start the application**:
   ```bash
   # Terminal 1 - Backend
   cd backend
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   python main.py
   
   # Terminal 2 - Frontend
   cd frontend
   npm start
   ```

5. **Access the application**:
   - Frontend: `http://localhost:3000`
   - Backend API docs: `http://localhost:8000/docs`

#### Option 2: Docker Deployment

4. **Start with Docker**:
   ```bash
   # Build and run all services
   docker-compose up --build
   
   # Or run in background
   docker-compose up -d
   ```

5. **Access the application**:
   - Frontend: `http://localhost:3000`
   - Backend API docs: `http://localhost:8000/docs`
   - ChromaDB: `http://localhost:8001`

### Manual Setup

If you prefer manual setup:

1. **Backend Setup**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   ```

3. **Create data directories**:
   ```bash
   mkdir -p data/{documents,confluence,models,chroma_db}
   ```

## Usage Guide

### 1. Document Upload and Analysis
- Navigate to the **Documents** page
- Upload PDF, DOCX, TXT, Markdown, or HTML files
- Wait for processing and indexing
- View document summaries and metadata

### 2. Chat with Documents
- Go to the **Chat** page
- Ask natural language questions about your documents
- Get AI-powered responses with source citations
- Maintain conversation context across queries

### 3. Confluence Integration
- Configure Confluence connection in **Settings**
- Import existing pages for context
- Generate new page drafts using AI assistance
- Export generated content to Confluence

### 4. Document Generation
- Use the chat interface to request new content
- Generate documents based on existing knowledge
- Create Confluence page templates
- Export in various formats

## Configuration

### Environment Variables

Create `backend/.env` with:

```env
# Backend Configuration
PORT=8000
HOST=0.0.0.0

# GPT4All Model Configuration
GPT4ALL_MODEL_PATH=../data/models
GPT4ALL_MODEL_NAME=nous-hermes-llama2-13b.Q4_0.gguf

# ChromaDB Configuration (for Docker)
CHROMA_HOST=chroma
CHROMA_PORT=8000

# Confluence Configuration (Optional)
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USERNAME=your-email@domain.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=YOUR_SPACE
```

### Recommended GPT4All Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `orca-mini-3b-gguf2-q4_0.gguf` | 1.9GB | Fast | Medium | Quick responses |
| `nous-hermes-llama2-13b.Q4_0.gguf` | 6.9GB | Medium | High | General use |
| `wizardlm-13b-v1.2.Q4_0.gguf` | 4.1GB | Medium | High | Complex reasoning |

## API Documentation

The backend provides a RESTful API with the following main endpoints:

- `POST /api/documents/upload` - Upload and process documents
- `GET /api/documents/` - List all documents
- `POST /api/chat/query` - Query documents with natural language
- `POST /api/chat/generate` - Generate new content
- `POST /api/confluence/import` - Import Confluence pages

Full API documentation is available at `http://localhost:8000/docs` when the backend is running.

## Development

### Running in Development Mode

```bash
# Backend with auto-reload
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend with hot reload
cd frontend
npm start
```

### Using Docker

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