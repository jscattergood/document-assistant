#!/bin/bash

echo "🚀 Setting up Document Assistant..."

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.9+ is required but not installed. Please install Python first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed. Please install Node.js first."
    exit 1
fi

# Create virtual environment for backend
echo "📦 Setting up Python virtual environment..."
cd backend
python3 -m venv venv

# Activate virtual environment and install dependencies
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/Linux/macOS
    source venv/bin/activate
fi

echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

cd ..

# Install frontend dependencies
echo "⚛️ Installing frontend dependencies..."
cd frontend
npm install

cd ..

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{documents,confluence,models,chroma_db}

# Download a sample GPT4All model (optional)
echo "🤖 GPT4All model setup..."
echo "To use offline AI capabilities, you'll need to download a GPT4All model."
echo "Recommended models:"
echo "  - nous-hermes-llama2-13b.Q4_0.gguf (4.1GB)"
echo "  - orca-mini-3b-gguf2-q4_0.gguf (1.9GB) - Faster, less capable"
echo ""
echo "Download a model from: https://gpt4all.io/index.html"
echo "Place the .gguf file in: ./data/models/"
echo ""

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > backend/.env << EOF
# Backend Configuration
PORT=8000
HOST=0.0.0.0

# GPT4All Model Configuration
GPT4ALL_MODEL_PATH=../data/models
GPT4ALL_MODEL_NAME=nous-hermes-llama2-13b.Q4_0.gguf

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=../data/chroma_db

# Document Storage
DOCUMENTS_DIRECTORY=../data/documents

# Logging
LOG_LEVEL=INFO

# Confluence Configuration (Optional - Configure these if you want Confluence integration)
# CONFLUENCE_URL=https://your-domain.atlassian.net
# CONFLUENCE_USERNAME=your-email@domain.com
# CONFLUENCE_API_TOKEN=your-api-token
# CONFLUENCE_SPACE_KEY=YOUR_SPACE
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Choose your deployment method:"
echo ""
echo "📋 Option 1: Local Development"
echo "1. Download a GPT4All model to ./data/models/ (see link above)"
echo "2. Start the backend: cd backend && source venv/bin/activate && python main.py"
echo "3. Start the frontend: cd frontend && npm start"
echo "4. Visit http://localhost:3000"
echo ""
echo "🐳 Option 2: Docker Deployment"
echo "1. Download a GPT4All model to ./data/models/ (see link above)"
echo "2. Run: docker-compose up --build"
echo "3. Visit http://localhost:3000"
echo ""
echo "📊 Port Configuration:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - API Documentation: http://localhost:8000/docs"
echo "  - ChromaDB (Docker only): http://localhost:8001"
echo ""
echo "📖 For more details, see README.md" 