@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  Aravalli Intelligence v1.0 — Startup Script (Windows)
REM  Team BIOBYTES — INNORAVE 2026 Eco-Hackathon
REM ─────────────────────────────────────────────────────────────────────────

echo.
echo   ========================================
echo     Aravalli Intelligence v1.0
echo     Ecological Monitoring System
echo     Team BIOBYTES
echo   ========================================
echo.

cd backend

echo [1/6] Installing Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

echo [2/6] Loading data...
python -m pipeline.ingest
if errorlevel 1 (
    echo [ERROR] Data ingestion failed
    pause
    exit /b 1
)
echo [OK] Data loaded

echo [3/6] Extracting features...
python -m pipeline.features
if errorlevel 1 (
    echo [ERROR] Feature engineering failed
    pause
    exit /b 1
)
echo [OK] Features extracted

echo [4/6] Running detection engine...
python -m pipeline.detect
if errorlevel 1 (
    echo [ERROR] Detection failed
    pause
    exit /b 1
)
echo [OK] Detection complete

echo [5/6] Generating AI reports...
python -m pipeline.explain
echo [OK] Reports generated

echo [6/6] Starting servers...
echo.
echo   API:  http://localhost:8000
echo   UI:   http://localhost:3000
echo   Docs: http://localhost:8000/docs
echo.

REM Start frontend in a new window
if exist ..\frontend\package.json (
    start cmd /k "cd ..\frontend && npm install && npm run dev"
    timeout /t 3 /nobreak >nul
)

REM Start FastAPI server (foreground)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
