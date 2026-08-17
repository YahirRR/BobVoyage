# BobVoyage — Space Weather Intelligence MCP Service

BobVoyage is an AI-agnostic space-weather intelligence platform that continuously
transforms real-time space-weather data into predictions, anomaly detection, event
correlations, and actionable spacecraft risk assessments.

It is consumed by IBM BOB, Claude, and any other MCP-compatible AI client via the
**Model Context Protocol (MCP)** — either locally over STDIO or remotely over
**Streamable HTTP**.

---

## Architecture

```
                         BobVoyage
                             │
                    ┌────────┴────────┐
                    │                 │
                  STDIO          Streamable HTTP
                    │                 │
                    ▼                 ▼
                   BOB        Remote MCP clients
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                        BOB       Claude      Other
```

The **same 8 tools** are exposed through both transports.  
No LLM dependency — BobVoyage is AI-agnostic.

---

## MCP Tools

| Tool | Description |
|---|---|
| `get_current_conditions` | Latest space-weather observation (NOAA / local) |
| `analyze_trends` | Directional trend analysis over a recent window |
| `detect_anomalies` | Z-score anomaly detection vs historical baseline |
| `predict_conditions` | Holt DES short-term forecast with ±1σ intervals |
| `assess_mission_risk` | 5-domain spacecraft risk assessment (OBSERVED/ANALYZED/PREDICTED/CORRELATED) |
| `correlate_space_events` | Temporal association between NASA DONKI events and NOAA telemetry |
| `recurrence_forecast` | Solar active-region recurrence projection (~27-day rotation) |
| `stakeholder_briefing` | Audience-specific plain-language risk briefing |

**MCP Prompt:** `bobvoyage_mission_context` — explains to any LLM how to use BobVoyage tools correctly, including evidence labels, causal-language constraints, and risk-score interpretation.

---

## Quick Start

### Prerequisites

```bash
pip install -e ".[dev]"
```

### Local STDIO (IBM BOB / development)

```bash
BOBVOYAGE_DATA_PROVIDER=local python -m bobvoyage.server
```

### Remote MCP — Streamable HTTP

```bash
# Demo mode (deterministic, no internet required)
BOBVOYAGE_DATA_PROVIDER=local \
uvicorn bobvoyage.mcp_http:app --host 0.0.0.0 --port 8080

# Live mode (NOAA + NASA DONKI)
BOBVOYAGE_DATA_PROVIDER=noaa \
BOBVOYAGE_NASA_API_KEY=<your-key> \
uvicorn bobvoyage.mcp_http:app --host 0.0.0.0 --port 8080
```

**MCP endpoint:** `http://localhost:8080/mcp`  
**Health check:** `http://localhost:8080/health`

### Mission Control Dashboard

```bash
BOBVOYAGE_DATA_PROVIDER=local \
uvicorn bobvoyage.dashboard.app:app --port 8080
# Open http://localhost:8080
```

### Docker

```bash
docker build -t bobvoyage .
docker run -p 8080:8080 -e BOBVOYAGE_DATA_PROVIDER=local bobvoyage
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BOBVOYAGE_DATA_PROVIDER` | `local` | `local` \| `noaa` \| `nasa_donki` |
| `BOBVOYAGE_NASA_API_KEY` | `DEMO_KEY` | NASA Open APIs key |
| `BOBVOYAGE_MCP_TRANSPORT` | `stdio` | `stdio` \| `http` |
| `BOBVOYAGE_MCP_HOST` | `0.0.0.0` | HTTP bind host |
| `BOBVOYAGE_MCP_PORT` | `8080` | HTTP bind port |
| `BOBVOYAGE_MCP_PATH` | `/mcp` | MCP endpoint path |
| `BOBVOYAGE_CORS_ORIGINS` | _(empty)_ | Comma-separated allowed CORS origins |

---

## Connecting an MCP-Compatible Client

Any client that supports the MCP Streamable HTTP transport can connect to:

```
https://your-domain.example/mcp
```

Example with the MCP Python SDK:

```python
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async with streamablehttp_client("http://localhost:8080/mcp") as (r, w, _):
    async with ClientSession(r, w) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("get_current_conditions_tool", {})
```

---

## Recommended Production Architecture

```
        Internet
            │
          HTTPS
            │
    Reverse Proxy / LB  ← authentication here
            │
            ▼
    BobVoyage HTTP MCP   (this process)
            │
    ┌───────┴────────┐
    ▼                ▼
 Providers       Intelligence
    │                │
NOAA / NASA    BobVoyage Tools
```

### Security Notes

- This server does **not** include authentication.  
  Place it behind an authenticated reverse proxy (nginx, Caddy, AWS ALB, etc.) for production.
- Serve via **HTTPS** in production.
- Never embed secrets in the Docker image or environment files committed to version control.
- API keys are read from environment variables — never logged or exposed through MCP responses.
- The CORS default is **empty** (no origins allowed) — set `BOBVOYAGE_CORS_ORIGINS` explicitly.

---

## Tests

```bash
BOBVOYAGE_DATA_PROVIDER=local python -m pytest tests/ -q
```

Current baseline: **684 tests, 0 failures**.

---

## Evidence Labels

Every BobVoyage tool output carries evidence labels:

| Label | Meaning |
|---|---|
| `OBSERVED` | Directly measured data from NOAA or local instruments |
| `ANALYZED` | Derived from observed data (z-scores, trends, statistics) |
| `PREDICTED` | Output of Holt DES forecasting model |
| `CORRELATED` | Temporal association with a space-weather event — **never** causation |
| `ASSESSED` | Mission-risk interpretation combining all evidence layers |

**Risk score ≠ failure probability.** BobVoyage provides decision-support intelligence for human operators.
