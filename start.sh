#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  Aravalli Intelligence v1.0 — Startup Script (macOS/Linux)
#  Team BIOBYTES — INNORAVE 2026 Eco-Hackathon
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log_info()  { echo -e "${GREEN}[✓]${NC} $1"; }
log_step()  { echo -e "${CYAN}[➤]${NC} ${BOLD}$1${NC}"; }
log_error() { echo -e "${RED}[✗]${NC} $1" >&2; }

echo ""
echo -e "${BOLD}  ╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}  ║      Aravalli Intelligence v1.0          ║${NC}"
echo -e "${BOLD}  ║      Ecological Monitoring System         ║${NC}"
echo -e "${BOLD}  ║      Team BIOBYTES                        ║${NC}"
echo -e "${BOLD}  ╚══════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Install Python dependencies
log_step "[1/6] Installing Python dependencies..."
if ! pip install -r requirements.txt --quiet 2>/dev/null; then
    log_error "Failed to install Python dependencies"
    exit 1
fi
log_info "Dependencies installed"

# Step 2: Data ingestion
log_step "[2/6] Loading data..."
if ! python -m pipeline.ingest; then
    log_error "Data ingestion failed"
    exit 1
fi
log_info "Data loaded"

# Step 3: Feature engineering
log_step "[3/6] Extracting features..."
if ! python -m pipeline.features; then
    log_error "Feature engineering failed"
    exit 1
fi
log_info "Features extracted"

# Step 4: Detection engine
log_step "[4/6] Running detection engine..."
if ! python -m pipeline.detect; then
    log_error "Detection failed"
    exit 1
fi
log_info "Detection complete"

# Step 5: Report generation
log_step "[5/6] Generating AI reports..."
if ! python -m pipeline.explain; then
    log_error "Report generation failed (non-critical)"
fi
log_info "Reports generated"

# Step 6: Start servers
log_step "[6/6] Starting servers..."
echo ""
echo -e "  ${GREEN}API:${NC}  http://localhost:8000"
echo -e "  ${GREEN}UI:${NC}   http://localhost:3000"
echo -e "  ${GREEN}Docs:${NC} http://localhost:8000/docs"
echo ""

# Start frontend dev server in background
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    cd frontend && npm install --silent && npm run dev &
    cd "$SCRIPT_DIR"
    sleep 2
fi

# Start FastAPI server (foreground)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
