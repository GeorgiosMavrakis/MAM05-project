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

**API Server:**
```bash
uvicorn API:app --host 0.0.0.0 --port 8000 --reload
```

**UI Server:**
```bash
cd rag-ui
npm run dev
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
| `vector_index.py` | Vector database utilities |
| `filter_whole.py` | Data cleaning and normalization |
| `chunker_whole.py` | Semantic text chunking |
| `embedder_whole.py` | GPU-accelerated embedding generation |
| `upload_to_qdrant.py` | Vector database indexing |

## Data Sources

- **Input**: FDA OpenFDA drug label dataset (`Drugs_filtered/`)
- **Combined Data**: `combined_whole.json` (all drug records)
- **Vector Index**: Qdrant remote instance

## Configuration

Key settings in `config.py`:
- `EMBED_MODEL`: Embedding model name (default: `all-MiniLM-L6-v2`)
- `QDRANT_URL`: Vector database endpoint
- `RETRIEVE_K_POOL`: Number of results to retrieve
- `EMBED_BATCH`: Batch size for embedding

## Frontend

Next.js-based React UI (`rag-ui/`) with:
- Responsive chat interface
- Thread management
- Markdown rendering
- Context display

## Development

For development and testing:
1. Run the API server with `--reload` flag for hot reloading
2. Run the UI server with `npm run dev`
3. Access UI at `http://localhost:3000`
4. API available at `http://localhost:8000`
