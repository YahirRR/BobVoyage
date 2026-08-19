# BobVoyage — Space Weather Intelligence MCP Service

**Challenge:** IBM AI Builders Challenge — August 2026 — Space Exploration Theme

BobVoyage is an AI-agnostic space-weather intelligence platform that continuously
transforms real-time space-weather data into predictions, anomaly detection, event
correlations, and actionable spacecraft risk assessments.

It is consumed by IBM BOB, Claude, and any other MCP-compatible AI client via the
**Model Context Protocol (MCP)** — either locally over STDIO or remotely over
**Streamable HTTP**.

---

## Problem Statement

Space weather events — solar flares, high speed solar wind streams, and geomagnetic
storms — happen constantly and are tracked in extraordinary detail by NASA and NOAA.
But that raw telemetry is not directly usable by the people who need to act on it.
A satellite operator, an astronaut, an airline, and a power grid operator all care
about the *same* solar event for completely different operational reasons, and today
each of them has to interpret the same dense scientific record themselves, with no
tooling that translates it into their specific context.

On top of that, solar active regions rotate with the Sun on a roughly 27-day cycle,
meaning a region that produced a severe flare is a real, physically grounded predictor
of near-term risk when it rotates back into an Earth-facing position — but this
recurrence pattern is invisible in a flat historical event log, and NOAA's own active
region numbering does not reliably track a region across a full rotation, making naive
"same ID reappears" approaches scientifically invalid (verified empirically against
this project's own 1,743-event dataset: zero instances of the same active region
reappearing within a 20–35 day window were found).

## Solution Description

BobVoyage ingests historical NASA/NOAA space weather data (1,743 events spanning
July 2023–July 2025: solar flares, high speed streams, and geomagnetic storms) plus
live NOAA/NASA DONKI telemetry, and exposes eight specialized intelligence tools over
the Model Context Protocol — anomaly detection, trend analysis, short-term forecasting,
event correlation, active-region recurrence projection, five-domain mission risk
scoring, and audience-specific stakeholder briefings.

Rather than hard-coding a chatbot around one specific LLM, BobVoyage is built
**AI-agnostic**: the intelligence lives in the MCP server itself, and any compatible
AI agent — IBM Bob, Claude, a local Ollama model, or any future MCP client — can
connect, discover the eight tools, and reason over real space weather data. This was
validated end-to-end with three independent clients: the official MCP Inspector, IBM
Bob (via `.bob/mcp.json`), and a fully local, free Ollama model (`qwen3:8b` via
`ollmcp`) — all successfully invoking `recurrence_forecast` and reasoning correctly
over its output.

The standout capability is **recurrence forecasting**: instead of relying on NOAA
active-region ID persistence (empirically shown not to hold in this dataset),
BobVoyage parses each region's real heliographic position (`source_location`, e.g.
`S25W90`) and projects its re-entry window using the Sun's actual synodic rotation
rate (~13.2°/day). Validated against a real historical case — active region 13664,
which produced 90 flares including an X8.7 in May 2024 — the tool correctly projects
a re-entry window of May 29–June 1, 2024, matching the documented "Mother's Day"
geomagnetic storm period.

A Mission Control dashboard provides a publicly demonstrable, no-install web interface
showing live risk scores, forecasts, event correlations, and a conversational assistant
("Ask BobVoyage") for judges and operators who want to see the system without
configuring an MCP client themselves.

## AI Approach and Architecture

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

## Selected Challenge Theme

**Space Exploration** — specifically: predictive spacecraft monitoring and anomaly
detection, space debris/event risk intelligence, and making space weather data
accessible and actionable across multiple real-world operational audiences.

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

BobVoyage has also been verified against local STDIO clients, including the official
MCP Inspector and [`ollmcp`](https://github.com/jonigl/mcp-client-for-ollama) (MCP
Client for Ollama), for teams who want to run a fully local, free AI agent against
these tools without any cloud dependency:

```bash
ollmcp -s "src/bobvoyage/server.py" -m qwen3:8b
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

---

## How IBM Bob Was Used

IBM Bob was used as the primary development tool throughout this project, both to
generate new capabilities and — critically — to diagnose and fix real bugs with
verifiable evidence rather than guesswork. A representative sample of concrete,
verifiable contributions:

- **Built `recurrence_forecast.py` from scratch**, including the scientific
  constraint analysis that led to rejecting the naive "same active-region ID
  reappears every 27 days" approach after Bob was directed to verify the assumption
  empirically against the real dataset (0 matches in a 20–35 day gap window across
  272 active regions) before writing any forecasting logic.
- **Built `stakeholder_briefing.py`**, translating `assess_mission_risk` output into
  audience-specific guidance for satellite operators, astronauts, aviation, and power
  grid operators.
- **Diagnosed and fixed a real production bug** where correlated-event badges in the
  dashboard displayed `"OTHER"` while the evidence text correctly showed `"CME"` or
  `"FLR"` — root-caused to `assess_mission_risk.py` reading a flat `event_type` key
  that did not exist, when the real value was nested at `corr["event"]["type"]`.
  Verified with 8 new regression tests and confirmed visually in the dashboard
  before merging.
- **Diagnosed and fixed a keyword substring collision** in `stakeholder_briefing.py`
  where the audience keyword `"eva"` (intended to match EVA/astronaut content) was
  matching inside the unrelated word `"Elevated"`, silently injecting incorrect
  power-domain recommendations into astronaut briefings. Found only after manually
  comparing chat output across all 4 audiences and noticing the recommendations were
  nearly identical — Bob then traced the root cause to two independent bugs (the
  substring collision and a "general recommendations" fallback that was being
  prepended unconditionally) and fixed both with a dedicated differentiation test suite.
- **Registered new tools into the MCP server** (`server.py`) following the existing
  decorator pattern, and diagnosed a dependency version-drift bug where an unpinned
  `mcp` package requirement silently upgraded to a breaking 2.0.0 release, changing
  the entire decorator API used throughout the server.
- **Authored `.bob/skills/bobvoyage-mcp-expert/SKILL.md`**, a project-specific agent
  skill encoding all of the above conventions and known failure modes, verified to
  activate automatically and correctly guide Bob's own behavior on new tasks without
  the conventions being restated manually.

---

## Team

- **Kevin Yahir Rojas Rodriguez** — Core architecture: MCP server, data providers
  (local/NOAA/NASA DONKI), Mission Control dashboard, Streamable HTTP transport,
  Docker deployment, and project documentation.
- **Sergio Alberto Hernández García** — MCP tools & intelligence layer: `recurrence_forecast`, `stakeholder_briefing`,
  event dataset integration, MCP server verification (Inspector, IBM Bob, Ollama),
  bug fixes (event badge mismatch, HSS event weighting), and the project's
  `bobvoyage-mcp-expert` agent skill.
