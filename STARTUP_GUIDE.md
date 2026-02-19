# Quick Start Guide - RAG System

## Prerequisites

Before running the system, ensure you have:

1. **Node.js** (v18 or higher)
   - Download from: https://nodejs.org/
   - Verify with: `node --version`

2. **Python** (3.10 or higher)
   - Download from: https://www.python.org/
   - Verify with: `python --version`

## Starting the System

### Windows (Easiest Method)

1. Navigate to the project folder: `MAM05-project`
2. Double-click `start_rag_system.bat`
3. Wait for the servers to start (about 10-30 seconds)
4. Your browser will automatically open to http://localhost:3000

### What the Script Does

The `start_rag_system.bat` file automatically:

1. ✅ Checks if Node.js is installed
2. ✅ Checks if Python is installed
3. ✅ Installs Python dependencies (`requirements.txt`)
4. ✅ Installs Node.js dependencies (`package.json`)
5. ✅ Starts the FastAPI backend server (port 8000)
6. ✅ Starts the Next.js frontend server (port 3000)
7. ✅ Opens your default browser to the application

### Manual Method

If you prefer to start servers manually:

**Terminal 1 - API Server:**
```bash
cd MAM05-project
uvicorn API:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - UI Server:**
```bash
cd MAM05-project/rag-ui
npm start
```

**Terminal 3 - Open Browser:**
```bash
start http://localhost:3000
```

## First Time Setup

The first run will take longer because:
- Python packages need to be downloaded (~500MB including PyTorch)
- Node.js packages need to be installed (~200MB)
- The Next.js production build needs to be compiled

Subsequent runs will be much faster (5-10 seconds).

## Accessing the Services

Once running, you can access:

- **UI Application**: http://localhost:3000
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Redoc**: http://localhost:8000/redoc

## Stopping the Servers

Two command windows will open when you run the script:
1. **RAG API Server** - FastAPI backend
2. **RAG UI Server** - Next.js frontend

To stop the servers:
- Close both command windows, OR
- Press `Ctrl+C` in each window

## Troubleshooting

### "Node.js is not installed"
- Install Node.js from https://nodejs.org/
- Restart your computer after installation
- Try running the script again

### "Python is not installed"
- Install Python from https://www.python.org/
- During installation, check "Add Python to PATH"
- Restart your computer after installation
- Try running the script again

### "Port already in use"
If you see errors about ports 3000 or 8000 being in use:
1. Close any other applications using these ports
2. Or modify the ports in the script:
   - API: Change `--port 8000` to another port
   - UI: Set `PORT=3001` before `npm start`

### "npm install failed"
- Make sure you have a stable internet connection
- Try running: `npm cache clean --force`
- Delete `rag-ui/node_modules` folder and try again

### "pip install failed"
- Make sure you have a stable internet connection
- Try running: `python -m pip install --upgrade pip`
- Install packages individually if needed

## System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for dependencies
- **Internet**: Required for first-time package installation
- **OS**: Windows 10/11

## Next Steps

Once the application is running:
1. Enter your pharmaceutical question in the chat interface
2. The system will retrieve relevant information from FDA drug labels
3. GPT-4 will generate a patient-friendly answer with citations
4. Citations link back to specific drug label sections

Example questions:
- "What are the side effects of aspirin?"
- "Can I take ibuprofen while pregnant?"
- "What is the recommended dosage for metformin?"

Enjoy using the RAG system!

