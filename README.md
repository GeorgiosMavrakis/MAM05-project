# MAM05-project

A Retrieval-Augmented Generation (RAG) system for pharmaceutical drug information, leveraging FDA drug labels and semantic search to provide accurate, cited answers about medication usage, side effects, and safety information.

## System Overview

This project implements a complete RAG pipeline:

```
FDA Data → Filter → Chunk → Embed → Vector DB (Qdrant) → Retrieve → Context Assembly → LLM → User
```

### Architecture

- **Backend API**: FastAPI application serving Q&A endpoints
- **Vector Database**: Qdrant instance for semantic search
- **Frontend UI**: Next.js React application for user interaction
- **Processing Pipeline**: Python modules for data ingestion and preparation

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js/npm or pnpm
- Qdrant server running (configured at `http://134.98.133.84:6333`)

### Starting the Servers

**Option 1: Install + Start (Windows)**

Use this the first time or whenever dependencies may be missing:
```bash
install_medication_assistant.bat
```

This script will:
- ✅ Check if Node.js and Python are installed
- ✅ Install all Python dependencies from `requirements.txt`
- ✅ Install all Node.js dependencies
- ✅ Start both API and UI servers
- ✅ Automatically open your browser to http://localhost:3000

**Option 1: Install + Start (macOS)**

Use this the first time or whenever dependencies may be missing. **Run these two commands in your terminal from the project folder:**

**Command 1:** Make the script executable
```bash
chmod +x ./install_medication_assistant_mac.sh
```

**Command 2:** Run the installation and startup script
```bash
./install_medication_assistant_mac.sh
```

This script will:
- ✅ Check if Node.js and Python are installed
- ✅ Install all Python dependencies from `requirements.txt`
- ✅ Install all Node.js dependencies
- ✅ Fix file permissions for Node.js binaries
- ✅ Start both API and UI servers (each in a new Terminal window)
- ✅ Automatically open your browser to http://localhost:3000

**Option 2: Start Only (Windows)**

Use this when dependencies are already installed:
```bash
start_medication_assistant.bat
```

This script will:
- ✅ Start both API and UI servers
- ✅ Automatically open your browser to http://localhost:3000

**Option 2: Start Only (macOS)**

Use this when dependencies are already installed. **Run these two commands in your terminal from the project folder:**

**Command 1:** Make the script executable
```bash
chmod +x ./start_medication_assistant_mac.sh
```

**Command 2:** Run the startup script
```bash
./start_medication_assistant_mac.sh
```

This script will:
- ✅ Fix file permissions for Node.js binaries
- ✅ Start both API and UI servers (each in a new Terminal window)
- ✅ Automatically open your browser to http://localhost:3000

**Option 3: Manual Startup**

**API Server:**
```bash
uvicorn API:app --host 0.0.0.0 --port 8000 --reload
```

**UI Server:**
```bash
cd rag-ui
npm start
```

## Data Processing Pipeline

The project follows a staged processing workflow using the `_whole` dataset variants (processing all drugs):

### Step 1: Filtering (`filter_whole.py`)
- Cleans and normalizes FDA OpenFDA JSON data
- Removes unwanted keys and table data
- Outputs: Filtered JSON ready for chunking

### Step 2: Chunking (`chunker_whole.py`)
- Transforms OpenFDA data into semantically coherent chunks
- Optimizes chunks for vector search with metadata filtering
- Input: `combined_whole.json` (all drugs from FDA dataset)
- Output: `chunks_whole.jsonl` (ready for embedding)

### Step 3: Embedding (`embedder_whole.py`)
- GPU-accelerated embedding using Sentence Transformers
- Model: `all-MiniLM-L6-v2`
- Parallel I/O processing for optimal speed
- Output: Embeddings ready for Qdrant indexing

### Step 4: Vector Database Upload (`upload_to_qdrant.py`)
- Indexes embeddings and chunks in Qdrant
- Enables semantic similarity search

## Runtime Pipeline

### Step 5: Retrieval (`retriever.py`)
- Embeds user queries using the same model
- Retrieves top-N semantically similar chunks from Qdrant
- Reranks results using cross-encoder
- Returns structured dictionary for downstream processing

### Step 6: Context Assembly & LLM Generation (`context_assembly.py`)
- Organizes retrieved chunks by drug and category
- Formats information hierarchically for LLM consumption
- Maintains source tracking for citations
- Generates patient-friendly answers with structured citations

### Step 7: API Endpoints (`API.py`)
- `/ask` - Synchronous Q&A endpoint
- `/stream` - Streaming Q&A endpoint
- Integrates retrieval and context assembly pipeline

## Core Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Central configuration (paths, API keys, thresholds, model names) |
| `retriever.py` | Query embedding, semantic search, and reranking |
| `context_assembly.py` | Context formatting and LLM integration |
| `API.py` | FastAPI application with Q&A endpoints |
| `filter_whole.py` | Data cleaning and normalization |
| `chunker_whole.py` | Semantic text chunking |
| `embedder_whole.py` | GPU-accelerated embedding generation |
| `upload_to_qdrant.py` | Vector database indexing |

## Data Sources

- **Input**: FDA OpenFDA drug label dataset (`Drugs_filtered/`)
- **Combined Data**: `combined_whole.json` (all drug records)
- **Vector Index**: Qdrant remote instance

## Frontend

Next.js-based React UI (`rag-ui/`) with:
- Responsive chat interface
- Thread management
- Markdown rendering
- Context display
- **Source references**: Displays vector IDs in an expandable section
  - After each assistant response, a "View Sources" expandable section appears
  - Sources shown as numbered list with vector IDs (e.g., `[1] ba4525b4-4d70-4eed-a195-b4bdaff99f23_precautions_7`)
  - Each source has a preview that can be expanded to see text excerpt
  - Uses HTML `<details>` tags for native expand/collapse functionality
  - Limited to top 10 most relevant sources
  - Clean markdown rendering without custom components

## Development

For development and testing:
1. Run the API server with `--reload` flag for hot reloading
2. Run the UI server with `npm run dev`
3. Access UI at `http://localhost:3000`
4. API available at `http://localhost:8000`
