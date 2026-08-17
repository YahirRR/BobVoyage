"""
BobVoyage — centralised environment-based configuration.

All runtime settings are read from environment variables here.
No other module should access os.environ for BobVoyage settings directly.

Environment variables
---------------------
BOBVOYAGE_DATA_PROVIDER   : "local" | "noaa" | "nasa_donki"  (default: "local")
BOBVOYAGE_NASA_API_KEY    : NASA Open APIs key                (default: "DEMO_KEY")
BOBVOYAGE_MCP_TRANSPORT   : "stdio" | "http"                  (default: "stdio")
BOBVOYAGE_MCP_HOST        : bind host for HTTP transport      (default: "0.0.0.0")
BOBVOYAGE_MCP_PORT        : bind port for HTTP transport      (default: 8080)
BOBVOYAGE_MCP_PATH        : MCP endpoint path                 (default: "/mcp")
BOBVOYAGE_CORS_ORIGINS    : comma-separated allowed origins   (default: "" → no CORS)
BOBVOYAGE_PORT            : alias for dashboard port          (default: 8080)
"""

from __future__ import annotations

import os


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Data provider
# ---------------------------------------------------------------------------
DATA_PROVIDER: str = _env("BOBVOYAGE_DATA_PROVIDER", "local").lower()
NASA_API_KEY:  str = _env("BOBVOYAGE_NASA_API_KEY",  "DEMO_KEY")

# ---------------------------------------------------------------------------
# MCP transport
# ---------------------------------------------------------------------------
MCP_TRANSPORT: str = _env("BOBVOYAGE_MCP_TRANSPORT", "stdio").lower()
MCP_HOST:      str = _env("BOBVOYAGE_MCP_HOST",      "0.0.0.0")
MCP_PORT:      int = _env_int("BOBVOYAGE_MCP_PORT",  8080)
MCP_PATH:      str = _env("BOBVOYAGE_MCP_PATH",      "/mcp")

# ---------------------------------------------------------------------------
# CORS (empty string → no CORS added)
# ---------------------------------------------------------------------------
_cors_raw: str = _env("BOBVOYAGE_CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = (
    [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if _cors_raw else []
)
