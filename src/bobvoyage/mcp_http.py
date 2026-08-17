"""
BobVoyage — Streamable HTTP MCP Server
=======================================

Exposes **all** BobVoyage intelligence tools over the MCP Streamable HTTP
transport using the official MCP SDK ``FastMCP`` API.

The same tool implementations used by the STDIO server are reused here
without modification.  The transport layer does **not** contain any
analytical logic.

Usage
-----
# Demo mode (deterministic, no internet required)
BOBVOYAGE_DATA_PROVIDER=local \\
uvicorn bobvoyage.mcp_http:app --host 0.0.0.0 --port 8080

# Live mode (NOAA + NASA DONKI)
BOBVOYAGE_DATA_PROVIDER=noaa \\
BOBVOYAGE_NASA_API_KEY=<your-key> \\
uvicorn bobvoyage.mcp_http:app --host 0.0.0.0 --port 8080

The MCP endpoint is available at:  POST/GET  /mcp

Health check (outside MCP protocol):  GET /health

Security note
-------------
This server does NOT implement authentication.  It is designed for
trusted/demo environments.  For production exposure, place this server
behind an authenticated reverse proxy (nginx, Caddy, AWS ALB, etc.) and
serve via HTTPS.  Never expose API keys through this interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from bobvoyage.config import MCP_PATH, CORS_ORIGINS
from bobvoyage.tools.current_conditions import get_current_conditions
from bobvoyage.tools.analyze_trends import analyze_trends
from bobvoyage.tools.detect_anomalies import detect_anomalies
from bobvoyage.tools.predict_conditions import predict_conditions
from bobvoyage.tools.assess_mission_risk import assess_mission_risk, DEFAULT_MISSION_PROFILE
from bobvoyage.tools.correlate_space_events import correlate_space_events
from bobvoyage.tools.recurrence_forecast import recurrence_forecast
from bobvoyage.tools.stakeholder_briefing import generate_stakeholder_briefing

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="BobVoyage",
    instructions=(
        "BobVoyage is a space-weather intelligence MCP service.  "
        "Use its tools to retrieve current space-weather conditions, "
        "analyse trends and anomalies, forecast parameters, assess "
        "mission risk, and identify temporal event correlations.  "
        "Always prefer calling tools over relying on training knowledge "
        "for space-weather data.  Evidence labels: OBSERVED = measured data; "
        "ANALYZED = derived statistical analysis; PREDICTED = forecast; "
        "CORRELATED = temporal association (never causal); "
        "ASSESSED = mission-risk interpretation.  "
        "Risk scores are decision-support values — not failure probabilities."
    ),
    stateless_http=True,          # each request is independent — no session state
    streamable_http_path=MCP_PATH,
)

# ---------------------------------------------------------------------------
# BobVoyage mission context prompt
# ---------------------------------------------------------------------------

@mcp.prompt()
def bobvoyage_mission_context() -> str:
    """
    BobVoyage mission intelligence context.

    Use this prompt when you need to understand how to interpret
    BobVoyage tool outputs correctly.
    """
    return """# BobVoyage — Space Weather Intelligence System

## What BobVoyage is
BobVoyage is an AI-agnostic space-weather intelligence MCP service.
It provides structured, traceable evidence about space-weather conditions
and their potential operational effects on spacecraft and infrastructure.

## Evidence Labels
Every piece of information carries an evidence label:

| Label       | Meaning                                                            |
|-------------|--------------------------------------------------------------------|
| OBSERVED    | Directly measured data from NOAA or local instruments              |
| ANALYZED    | Derived from observed data (z-scores, trends, statistics)          |
| PREDICTED   | Output of Holt DES forecasting model — short-term projection only  |
| CORRELATED  | Temporal association with a space-weather event (NOT causation)    |
| ASSESSED    | Mission-risk interpretation combining all evidence layers          |

## Critical constraints
- CORRELATED never means CAUSED. Two events close in time are
  "temporally associated" or "potentially related" — never proven causal.
- Risk scores are decision-support values. They are NOT failure probabilities.
- Missing or None values mean the provider cannot supply that measurement.
  Never invent or substitute data.
- Stale data must be disclosed to the operator.

## Tool usage guide

### Start with current conditions
Call `get_current_conditions` first to establish the baseline state.

### Build the intelligence pipeline
1. `get_current_conditions`   → current observed values
2. `analyze_trends`           → recent directional changes
3. `detect_anomalies`         → statistical deviations from baseline
4. `predict_conditions`       → short-term forecast (Holt DES)
5. `correlate_space_events`   → temporal event associations
6. `assess_mission_risk`      → combined mission-risk assessment
7. `stakeholder_briefing`     → audience-specific narrative
8. `recurrence_forecast`      → active-region recurrence projection

### Tools are composable
Pass the output of earlier tools into later tools.
Example: pass `get_current_conditions` output into `assess_mission_risk`.

## Domains assessed
| Domain            | Affected by                              |
|-------------------|------------------------------------------|
| radiation         | Proton flux, SEP events, CMEs            |
| communications    | X-ray flux, geomagnetic index, CME, FLR  |
| navigation        | Geomagnetic index, CME, GST, HSS         |
| power             | Proton flux, SEP, geomagnetic induction  |
| attitude_control  | Solar wind pressure, magnetic field, GST |

## Risk levels
LOW → MODERATE → HIGH → CRITICAL

These are operational thresholds for decision support.
They do NOT represent scientific probability of failure.
"""

# ---------------------------------------------------------------------------
# Tool registrations — one implementation per tool, shared with STDIO server
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Retrieve the latest space-weather observation.\n\n"
        "WHEN TO CALL: Whenever the user asks about current conditions, "
        "the live environment, or before running any analysis.\n\n"
        "RETURNS: Timestamp, solar wind speed/density, magnetic field, "
        "X-ray flux, proton flux, geomagnetic Kp index, data age, "
        "staleness flag, and provider source.\n\n"
        "LIMITATIONS: If is_stale is true, the observation predates the "
        "provider's freshness threshold — disclose this to the operator."
    )
)
def get_current_conditions_tool(dataset_path: str | None = None) -> dict[str, Any]:
    """Get the most recent space-weather observation."""
    return get_current_conditions(dataset_path=dataset_path)


@mcp.tool(
    description=(
        "Analyse recent space-weather trends.\n\n"
        "WHEN TO CALL: When the user asks what has changed, whether "
        "conditions are worsening or improving, or before assess_mission_risk.\n\n"
        "RETURNS: Per-parameter direction (increasing/decreasing/stable), "
        "change %, severity (stable/minor/moderate/significant), and "
        "a ranked list of significant trends.\n\n"
        "OUTPUT TYPE: ANALYZED evidence."
    )
)
def analyze_trends_tool(window: int = 12, dataset_path: str | None = None) -> dict[str, Any]:
    """Analyse directional trends over a recent observation window."""
    return analyze_trends(window=window, dataset_path=dataset_path)


@mcp.tool(
    description=(
        "Detect statistical anomalies in space-weather observations.\n\n"
        "WHEN TO CALL: When the user asks whether anything is unusual, "
        "or before assess_mission_risk for anomaly evidence.\n\n"
        "METHOD: Z-score comparison of a recent window against a "
        "historical baseline.  Flags moderate (|z|≥2) and "
        "significant (|z|≥3) deviations.\n\n"
        "OUTPUT TYPE: ANALYZED evidence."
    )
)
def detect_anomalies_tool(
    recent_window: int = 6,
    baseline_window: int = 48,
    z_threshold: float = 2.0,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """Detect statistically significant deviations from baseline."""
    return detect_anomalies(
        recent_window=recent_window,
        baseline_window=baseline_window,
        z_threshold=z_threshold,
        dataset_path=dataset_path,
    )


@mcp.tool(
    description=(
        "Forecast space-weather parameters.\n\n"
        "WHEN TO CALL: When the user asks what is expected next, "
        "or to provide predicted evidence to assess_mission_risk.\n\n"
        "METHOD: Holt Double Exponential Smoothing with walk-forward "
        "validation.  Returns predicted values, ±1σ prediction intervals, "
        "MAE/RMSE, and a reliability score.\n\n"
        "IMPORTANT: Predicted values are short-term statistical projections. "
        "They are NOT guaranteed outcomes. Always disclose the horizon.\n\n"
        "OUTPUT TYPE: PREDICTED evidence."
    )
)
def predict_conditions_tool(
    horizon: int = 12,
    lookback: int = 48,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """Forecast space-weather parameters using Holt DES."""
    return predict_conditions(horizon=horizon, lookback=lookback, dataset_path=dataset_path)


@mcp.tool(
    description=(
        "Assess spacecraft operational risk from space-weather evidence.\n\n"
        "WHEN TO CALL: When the user asks whether current conditions "
        "may affect spacecraft or infrastructure operations.\n\n"
        "INPUTS: Pass outputs from get_current_conditions (conditions), "
        "analyze_trends (trends), detect_anomalies (anomalies), "
        "predict_conditions (predictions), correlate_space_events "
        "(correlated_events).  All inputs are optional — the tool "
        "adapts to available evidence.\n\n"
        "RETURNS: 5-domain risk scores (radiation, communications, "
        "navigation, power, attitude_control), overall risk level "
        "(LOW/MODERATE/HIGH/CRITICAL), traceable evidence, and "
        "domain-specific recommendations.\n\n"
        "IMPORTANT: Risk score ≠ failure probability. "
        "This is decision-support intelligence."
    )
)
def assess_mission_risk_tool(
    conditions: dict | None = None,
    trends: dict | None = None,
    anomalies: list | None = None,
    predictions: list | None = None,
    correlated_events: list | None = None,
    mission_profile: dict | None = None,
) -> dict[str, Any]:
    """Assess mission risk across 5 spacecraft domains."""
    return assess_mission_risk(
        conditions=conditions,
        trends=trends,
        anomalies=anomalies,
        predictions=predictions,
        correlated_events=correlated_events,
        mission_profile=mission_profile,
    )


@mcp.tool(
    description=(
        "Identify temporal associations between space-weather events "
        "and telemetry observations.\n\n"
        "WHEN TO CALL: When you have NASA DONKI events (CME, FLR, GST, "
        "SEP, HSS) and want to check whether they temporally coincide "
        "with observed telemetry anomalies.\n\n"
        "OUTPUT: Per-event correlation scores (0–1), interpretation "
        "(no_significant / weak / moderate / strong temporal association), "
        "component breakdown, and evidence strings.\n\n"
        "CRITICAL: Correlation is NOT causation. Never claim an event "
        "caused an observation. Use 'temporally associated with' language."
    )
)
def correlate_space_events_tool(
    events: list | None = None,
    observations: list | None = None,
    lookback_hours: float = 4.0,
    lookahead_hours: float = 2.0,
    min_score: float = 0.1,
) -> dict[str, Any]:
    """Identify temporal associations between events and observations."""
    return correlate_space_events(
        events=events,
        observations=observations,
        lookback_hours=lookback_hours,
        lookahead_hours=lookahead_hours,
        min_score=min_score,
    )


@mcp.tool(
    description=(
        "Forecast recurrence risk for solar active regions.\n\n"
        "WHEN TO CALL: When the user asks about active regions rotating "
        "back into Earth view, future flare risk from known active regions, "
        "or the ~27-day solar rotation recurrence cycle.\n\n"
        "METHOD: Physics-based heliographic position tracking using the "
        "Sun's ~27-day synodic rotation rate.  NOT a machine-learning model.\n\n"
        "LIMITATIONS: Active region NOAA numbers are not reliable "
        "persistence trackers across rotations. Report as "
        "low/moderate/elevated risk only — never as certainty."
    )
)
def recurrence_forecast_tool(
    active_region: int | None = None,
    lookback_days: float = 45.0,
    as_of: str | None = None,
    min_flares: int = 2,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """Forecast solar active region recurrence risk."""
    return recurrence_forecast(
        active_region=active_region,
        lookback_days=lookback_days,
        as_of=as_of,
        min_flares=min_flares,
        dataset_path=dataset_path,
    )


@mcp.tool(
    description=(
        "Translate mission risk into audience-specific plain-language briefings.\n\n"
        "WHEN TO CALL: When the user wants a human-readable summary for "
        "a specific operational community.\n\n"
        "AUDIENCES: satellite_operator, astronaut, aviation, power_grid.\n\n"
        "INPUT: Pass the complete output of assess_mission_risk() as "
        "risk_assessment.\n\n"
        "RETURNS: Filtered domains relevant to the audience, plain-English "
        "risk narrative, matching recommendations, and evidence provenance.\n\n"
        "No new risk values are computed — this is translation only."
    )
)
def stakeholder_briefing_tool(
    risk_assessment: dict | None = None,
    audience: str | None = None,
) -> dict[str, Any]:
    """Generate an audience-specific plain-language risk briefing."""
    return generate_stakeholder_briefing(
        risk_assessment=risk_assessment,
        audience=audience,
    )


# ---------------------------------------------------------------------------
# Starlette ASGI application
# ---------------------------------------------------------------------------
# Compose: MCP at MCP_PATH + /health outside the MCP protocol.

async def health(request: Request) -> JSONResponse:
    """Lightweight health check for load-balancers and deployment probes."""
    return JSONResponse({
        "status": "ok",
        "service": "bobvoyage",
        "mcp": "available",
        "endpoint": MCP_PATH,
        "data_provider": os.environ.get("BOBVOYAGE_DATA_PROVIDER", "local"),
    })


def _build_app() -> Starlette:
    """Build the ASGI application, optionally adding CORS middleware."""
    mcp_asgi = mcp.streamable_http_app()

    routes = [
        Route("/health", health),
        Mount(MCP_PATH, app=mcp_asgi),
    ]

    starlette_app = Starlette(routes=routes)

    if CORS_ORIGINS:
        from starlette.middleware.cors import CORSMiddleware
        starlette_app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    return starlette_app


app = _build_app()

# ---------------------------------------------------------------------------
# Direct run entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    from bobvoyage.config import MCP_HOST, MCP_PORT

    provider = os.environ.get("BOBVOYAGE_DATA_PROVIDER", "local").upper()
    print(f"BobVoyage MCP HTTP server — {provider} mode")
    print(f"MCP endpoint : http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}")
    print(f"Health check : http://{MCP_HOST}:{MCP_PORT}/health")
    print("⚠  No authentication — use behind a trusted reverse proxy for production.")
    uvicorn.run("bobvoyage.mcp_http:app", host=MCP_HOST, port=MCP_PORT, reload=False)
