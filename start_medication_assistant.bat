@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Start Medication Assistant (No Install)
echo ========================================
echo.

REM Get the directory where the batch file is located
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [1/2] Starting API Server...
echo Starting FastAPI on http://localhost:8000
start "RAG API Server" cmd /k "cd /d "%PROJECT_DIR%" && echo Starting API Server... && uvicorn API:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

echo [2/2] Starting UI Server...
echo Starting Next.js UI on http://localhost:3000
start "RAG UI Server" cmd /k "cd /d "%PROJECT_DIR%\rag-ui" && echo Starting UI Server... && npm start"

echo ========================================
echo Waiting for servers to start...
echo ========================================
timeout /t 5 /nobreak >nul

echo Opening browser...
start http://localhost:3000

echo.
echo ========================================
echo RAG System Started Successfully!
echo ========================================
echo.
echo API Server:  http://localhost:8000
echo UI Server:   http://localhost:3000
echo API Docs:    http://localhost:8000/docs
echo.
echo Two command windows have been opened:
echo  - RAG API Server (FastAPI with uvicorn)
echo  - RAG UI Server (Next.js)
echo.
echo Close those windows to stop the servers.
echo Press any key to close this window...
pause >nul
