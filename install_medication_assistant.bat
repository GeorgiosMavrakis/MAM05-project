@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Start Medication Assistant
echo ========================================
echo.

REM Get the directory where the batch file is located
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [1/6] Checking Node.js installation...
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)
node --version
echo Node.js found!
echo.

echo [2/6] Checking Python installation...
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.10+ from https://www.python.org/
    echo.
    pause
    exit /b 1
)
python --version
echo Python found!
echo.

echo [2.5/6] Checking Rust/Cargo installation...
where cargo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Rust/Cargo is not installed!
    echo Attempting to install Rust automatically...
    echo.
    powershell -Command "Invoke-WebRequest -Uri 'https://win.rustup.rs/x86_64' -OutFile $env:USERPROFILE\rustup-init.exe"
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to download Rust installer!
        echo Please install Rust manually from https://rustup.rs/
        echo.
        pause
        exit /b 1
    )
    echo Running Rust installer...
    "%USERPROFILE%\rustup-init.exe" -y
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install Rust!
        echo.
        pause
        exit /b 1
    )
    echo Please restart your terminal after Rust installation.
    echo Rust has been installed, re-run this batch file.
    echo.
    pause
    exit /b 1
)
cargo --version
rustc --version
echo Rust/Cargo found!
echo.

echo [3/6] Installing/Checking Python dependencies...
if not exist "requirements.txt" (
    echo WARNING: requirements.txt not found. Skipping Python dependency check.
) else (
    echo Installing Python packages from requirements.txt...
    echo Upgrading pip, setuptools, and wheel...
    python -m pip install --upgrade pip setuptools wheel
    echo Installing dependencies with binary wheels...
    python -m pip install --only-binary=:all: -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install Python dependencies!
        echo Attempting standard installation as fallback...
        python -m pip install -r requirements.txt
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to install Python dependencies!
            pause
            exit /b 1
        )
    )
    echo Python dependencies installed successfully!
)
echo.

echo [4/6] Installing/Checking Node.js dependencies...
cd rag-ui
if not exist "node_modules" (
    echo node_modules not found. Running npm install...
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install Node.js dependencies!
        cd ..
        pause
        exit /b 1
    )
) else (
    echo node_modules found. Checking for updates...
    call npm install
)
echo Node.js dependencies ready!
cd ..
echo.

echo [5/6] Starting API Server...
echo Starting FastAPI on http://localhost:8000
start "RAG API Server" cmd /k "cd /d "%PROJECT_DIR%" && echo Starting API Server... && python -m uvicorn API:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul
echo API Server started!
echo.

echo [6/6] Starting UI Server...
echo Starting Next.js UI on http://localhost:3000
start "RAG UI Server" cmd /k "cd /d "%PROJECT_DIR%\rag-ui" && echo Starting UI Server... && npm start"
echo UI Server starting...
echo.

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

