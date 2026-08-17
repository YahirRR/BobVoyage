#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BobVoyage — MCP HTTP only (for connecting to Claude.ai / IBM Bob / any LLM)
#
# Usage:
#   bash scripts/start_mcp_only.sh              # demo mode, port 8090
#   PROVIDER=noaa bash scripts/start_mcp_only.sh  # live NOAA data
#   PORT=8080 bash scripts/start_mcp_only.sh       # custom port
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROVIDER="${PROVIDER:-local}"
PORT="${PORT:-8090}"

GRN='\033[0;32m'
BLU='\033[0;34m'
YLW='\033[0;33m'
RST='\033[0m'

echo ""
echo -e "${BLU}BobVoyage MCP HTTP Server${RST}"
echo -e "  Provider : ${YLW}${PROVIDER}${RST}"
echo -e "  MCP URL  : ${GRN}http://localhost:${PORT}/mcp${RST}"
echo -e "  Health   : ${GRN}http://localhost:${PORT}/health${RST}"
echo ""
echo "Configure your MCP client with:"
echo "  URL       : http://localhost:${PORT}/mcp"
echo "  Transport : streamable-http"
echo ""

BOBVOYAGE_DATA_PROVIDER="$PROVIDER" \
BOBVOYAGE_MCP_PORT="$PORT" \
  python3 -m uvicorn bobvoyage.mcp_http:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info
