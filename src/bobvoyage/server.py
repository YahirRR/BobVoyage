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
        )
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
