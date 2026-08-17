#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BobVoyage — Demo Launcher
#
# Starts both servers on separate ports:
#   - Dashboard (UI + /api/ask chat)  →  http://localhost:8080
#   - MCP HTTP  (Streamable HTTP MCP) →  http://localhost:8090/mcp
#
# Usage:
#   bash scripts/start_demo.sh              # demo mode (local CSV, no internet)
#   PROVIDER=noaa bash scripts/start_demo.sh  # live mode (NOAA + NASA DONKI)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROVIDER="${PROVIDER:-local}"
DASH_PORT="${DASH_PORT:-8080}"
MCP_PORT="${MCP_PORT:-8090}"

# Colours
GRN='\033[0;32m'
BLU='\033[0;34m'
YLW='\033[0;33m'
RST='\033[0m'

echo ""
echo -e "${BLU}╔══════════════════════════════════════════════════════╗${RST}"
echo -e "${BLU}║         BobVoyage — Mission Control                  ║${RST}"
echo -e "${BLU}╚══════════════════════════════════════════════════════╝${RST}"
echo ""
echo -e "  Provider  : ${YLW}${PROVIDER}${RST}"
echo -e "  Dashboard : ${GRN}http://localhost:${DASH_PORT}${RST}"
echo -e "  MCP HTTP  : ${GRN}http://localhost:${MCP_PORT}/mcp${RST}"
echo -e "  Health    : ${GRN}http://localhost:${MCP_PORT}/health${RST}"
echo ""
echo -e "  ${YLW}Press Ctrl+C to stop both servers.${RST}"
echo ""

# Trap to kill both background processes on exit
cleanup() {
  echo ""
  echo "Stopping BobVoyage servers…"
  kill "$DASH_PID" "$MCP_PID" 2>/dev/null || true
  wait "$DASH_PID" "$MCP_PID" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# ── Dashboard ────────────────────────────────────────────────────────────────
BOBVOYAGE_DATA_PROVIDER="$PROVIDER" \
BOBVOYAGE_PORT="$DASH_PORT" \
  python3 -m uvicorn bobvoyage.dashboard.app:app \
    --host 0.0.0.0 \
    --port "$DASH_PORT" \
    --workers 1 \
    --log-level warning &
DASH_PID=$!

# ── MCP HTTP server ──────────────────────────────────────────────────────────
BOBVOYAGE_DATA_PROVIDER="$PROVIDER" \
BOBVOYAGE_MCP_PORT="$MCP_PORT" \
  python3 -m uvicorn bobvoyage.mcp_http:app \
    --host 0.0.0.0 \
    --port "$MCP_PORT" \
    --workers 1 \
    --log-level warning &
MCP_PID=$!

# Wait for both
wait "$DASH_PID" "$MCP_PID"
