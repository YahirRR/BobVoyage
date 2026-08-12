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
