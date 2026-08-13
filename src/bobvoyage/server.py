"""
BobVoyage MCP Server — entry point

Exposes BobVoyage space-weather tools over the MCP stdio transport.
Start with: python -m bobvoyage.server
"""

from __future__ import annotations

import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from bobvoyage.tools.current_conditions import get_current_conditions
from bobvoyage.tools.analyze_trends import analyze_trends
from bobvoyage.tools.detect_anomalies import detect_anomalies
from bobvoyage.tools.predict_conditions import predict_conditions
from bobvoyage.tools.assess_mission_risk import assess_mission_risk, DEFAULT_MISSION_PROFILE
from bobvoyage.tools.correlate_space_events import correlate_space_events
from bobvoyage.tools.recurrence_forecast import recurrence_forecast

# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------
app = Server("bobvoyage")

# ---------------------------------------------------------------------------
# Tool: list_tools
# ---------------------------------------------------------------------------
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_current_conditions",
            description=(
                "Retrieve the most recent space-weather observation from the local "
                "development dataset. Returns timestamp, solar wind speed/density, "
                "magnetic field, X-ray flux, proton flux, and geomagnetic index. "
                "Data retrieval only — no prediction or risk assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": (
                            "Optional absolute or relative path to the CSV dataset. "
                            "Defaults to data/space_weather.csv in the project root."
                        ),
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="analyze_trends",
            description=(
                "Analyze recent space-weather observations and identify meaningful "
                "trends. Returns per-parameter direction (increasing/decreasing/stable), "
                "percentage and absolute change, severity classification, and a ranked "
                "list of significant trends. Descriptive analysis only — no prediction "
                "or risk assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "window": {
                        "type": "integer",
                        "description": (
                            "Number of most-recent observations to include. "
                            "Must be ≥ 2. Default is 12 (≈ 1 hour at 5-min resolution)."
                        ),
                        "default": 12,
                        "minimum": 2,
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": (
                            "Optional path to the CSV dataset. "
                            "Defaults to data/space_weather.csv in the project root."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="detect_anomalies",
            description=(
                "Identify space-weather observations that significantly deviate from "
                "a historical baseline using z-score analysis. Returns per-parameter "
                "anomaly details including observed value, baseline mean/std, z-score, "
                "severity (moderate/significant), and direction. Statistical anomaly "
                "detection only — no prediction or risk assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "recent_window": {
                        "type": "integer",
                        "description": (
                            "Number of latest observations to examine. "
                            "Must be ≥ 1. Default is 6."
                        ),
                        "default": 6,
                        "minimum": 1,
                    },
                    "baseline_window": {
                        "type": "integer",
                        "description": (
                            "Number of observations before the recent window used to "
                            "establish the baseline (mean, std). Must be ≥ 3. Default is 48."
                        ),
                        "default": 48,
                        "minimum": 3,
                    },
                    "z_threshold": {
                        "type": "number",
                        "description": (
                            "Minimum |z-score| to flag an observation as anomalous. "
                            "Must be > 0. Default is 2.0."
                        ),
                        "default": 2.0,
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": (
                            "Optional path to the CSV dataset. "
                            "Defaults to data/space_weather.csv in the project root."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="predict_conditions",
            description=(
                "Forecast space-weather parameters using Holt's Double Exponential "
                "Smoothing. Returns per-parameter predicted values with prediction "
                "intervals, walk-forward validation metrics (MAE, RMSE, MAPE), and "
                "a confidence score. Forecasting only — no anomaly detection or "
                "risk assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "horizon": {
                        "type": "integer",
                        "description": (
                            "Number of future steps to forecast. Must be ≥ 1. "
                            "Default 12. At 5-min resolution: 12 → 60 min."
                        ),
                        "default": 12,
                        "minimum": 1,
                    },
                    "lookback": {
                        "type": "integer",
                        "description": (
                            "Most-recent historical observations used for fitting. "
                            "Must be ≥ 10. Default 48 (≈ 4 hours)."
                        ),
                        "default": 48,
                        "minimum": 10,
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": (
                            "Optional path to the CSV dataset. "
                            "Defaults to data/space_weather.csv in the project root."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="assess_mission_risk",
            description=(
                "Assess spacecraft operational risk by combining current space-weather "
                "observations, trend analysis, anomaly detection, short-term forecasts, "
                "and optionally correlated space-weather events (M8). Returns domain-level "
                "risk scores (radiation, communications, navigation, power, attitude_control), "
                "an overall mission risk level (LOW/MODERATE/HIGH/CRITICAL), traceable "
                "evidence (OBSERVED/ANALYZED/PREDICTED/CORRELATED), and recommendations. "
                "Accepts a configurable mission sensitivity profile."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "object",
                        "description": (
                            "Observation dict from get_current_conditions()[\"observation\"]. "
                            "Omit to assess without current observations."
                        ),
                    },
                    "trends": {
                        "type": "object",
                        "description": (
                            "Trend dict from analyze_trends()[\"trends\"]. "
                            "Omit to assess without trend analysis."
                        ),
                    },
                    "anomalies": {
                        "type": "array",
                        "description": (
                            "Anomaly list from detect_anomalies()[\"anomalies\"]. "
                            "Omit or pass [] when no anomalies."
                        ),
                    },
                    "predictions": {
                        "type": "array",
                        "description": (
                            "Prediction list from predict_conditions()[\"predictions\"]. "
                            "Omit to assess without forecast data."
                        ),
                    },
                    "correlated_events": {
                        "type": "array",
                        "description": (
                            "Correlation list from correlate_space_events()[\"correlations\"]. "
                            "When supplied, each event's correlation score is translated into "
                            "a domain-specific risk addend via the event-to-domain relevance "
                            "matrix.  An env-saturation discount prevents double-counting with "
                            "environmental evidence."
                        ),
                    },
                    "mission_profile": {
                        "type": "object",
                        "description": (
                            "Sensitivity profile with keys: radiation_sensitivity, "
                            "communications_sensitivity, navigation_sensitivity, "
                            "power_sensitivity, attitude_control_sensitivity. "
                            "Each value: 'low', 'medium', or 'high'. "
                            f"Defaults to: {DEFAULT_MISSION_PROFILE}."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="correlate_space_events",
            description=(
                "Identify temporal associations between NASA DONKI space-weather "
                "events (CME, flare, geomagnetic storm, SEP) and NOAA telemetry "
                "observations. Returns per-event correlation scores, statistical "
                "deviations, component breakdown, and guarded evidence strings. "
                "Establishes statistical/temporal association only — never claims "
                "causality. Intelligence layer consumed by mission risk assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "description": (
                            "List of SpaceWeatherEvent dicts from NASA DONKI provider. "
                            "Each must have event_type and event_time."
                        ),
                    },
                    "observations": {
                        "type": "array",
                        "description": (
                            "List of SpaceWeatherObservation dicts from NOAA provider. "
                            "Each must have timestamp and at least one numeric parameter."
                        ),
                    },
                    "lookback_hours": {
                        "type": "number",
                        "description": "Hours before each event to search for correlated observations.",
                        "default": 4.0,
                    },
                    "lookahead_hours": {
                        "type": "number",
                        "description": "Hours after each event to search for correlated observations.",
                        "default": 2.0,
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum correlation score [0,1] to include in results.",
                        "default": 0.1,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="recurrence_forecast",
            description=(
                "Forecast recurrence risk for solar active regions using heliographic "
                "position (source_location) and the Sun's synodic rotation rate. "
                "Identifies regions with a history of strong flare production that are "
                "approaching or past the west limb, and projects a re-entry window when "
                "they may rotate back into an Earth-facing position (~27 days later). "
                "This is a physics-based projection, not a guaranteed prediction — active "
                "region NOAA numbers are not reliable persistence trackers, so recurrence "
                "risk is reported as low/moderate/elevated, never as a certain event."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "active_region": {
                        "type": "integer",
                        "description": (
                            "Optional. Restrict the forecast to a specific active region "
                            "number. Omit to scan all regions active within lookback_days."
                        ),
                    },
                    "lookback_days": {
                        "type": "number",
                        "description": (
                            "Only consider regions whose most recent flare falls within "
                            "this many days of 'as_of'. Default 45."
                        ),
                        "default": 45.0,
                    },
                    "as_of": {
                        "type": "string",
                        "description": (
                            "Optional ISO-8601 timestamp to forecast from. Defaults to "
                            "the most recent flare timestamp in the dataset."
                        ),
                    },
                    "min_flares": {
                        "type": "integer",
                        "description": (
                            "Minimum flares a region must have to appear in a full scan. "
                            "Ignored when active_region is specified. Default 2."
                        ),
                        "default": 2,
                        "minimum": 1,
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": (
                            "Optional path to the CSV dataset. "
                            "Defaults to data/space_weather_unified.csv in the project root."
                        ),
                    },
                },
                "required": [],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool: call_tool
# ---------------------------------------------------------------------------
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_current_conditions":
        dataset_path = arguments.get("dataset_path")
        result = get_current_conditions(dataset_path=dataset_path)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "analyze_trends":
        window = arguments.get("window", 12)
        dataset_path = arguments.get("dataset_path")
        result = analyze_trends(window=window, dataset_path=dataset_path)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "detect_anomalies":
        recent_window   = arguments.get("recent_window", 6)
        baseline_window = arguments.get("baseline_window", 48)
        z_threshold     = arguments.get("z_threshold", 2.0)
        dataset_path    = arguments.get("dataset_path")
        result = detect_anomalies(
            recent_window=recent_window,
            baseline_window=baseline_window,
            z_threshold=z_threshold,
            dataset_path=dataset_path,
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "predict_conditions":
        horizon      = arguments.get("horizon", 12)
        lookback     = arguments.get("lookback", 48)
        dataset_path = arguments.get("dataset_path")
        result = predict_conditions(
            horizon=horizon,
            lookback=lookback,
            dataset_path=dataset_path,
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "assess_mission_risk":
        result = assess_mission_risk(
            conditions         = arguments.get("conditions"),
            trends             = arguments.get("trends"),
            anomalies          = arguments.get("anomalies"),
            predictions        = arguments.get("predictions"),
            correlated_events  = arguments.get("correlated_events"),
            mission_profile    = arguments.get("mission_profile"),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "correlate_space_events":
        result = correlate_space_events(
            events            = arguments.get("events"),
            observations      = arguments.get("observations"),
            lookback_hours    = arguments.get("lookback_hours", 4.0),
            lookahead_hours   = arguments.get("lookahead_hours", 2.0),
            min_score         = arguments.get("min_score", 0.1),
        )
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "recurrence_forecast":
        result = recurrence_forecast(
            active_region  = arguments.get("active_region"),
            lookback_days  = arguments.get("lookback_days", 45.0),
            as_of          = arguments.get("as_of"),
            dataset_path   = arguments.get("dataset_path"),
            min_flares     = arguments.get("min_flares", 2),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    return [
        TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"Unknown tool: '{name}'"}),
        )
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
