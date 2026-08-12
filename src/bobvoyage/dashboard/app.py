"""
BobVoyage Dashboard — FastAPI Application

Serves:
  GET  /               → dashboard.html
  GET  /api/status     → health check
  GET  /api/assessment → full pipeline assessment (demo or live)
  POST /api/assess     → re-assess with custom mission profile
  GET  /api/forecast   → single-parameter forecast data
  POST /api/ask        → conversational question routing

Start with:
  uvicorn bobvoyage.dashboard.app:app --reload --port 8080
or:
  python -m bobvoyage.dashboard.app
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bobvoyage.dashboard.service import get_assessment, ask_question

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BobVoyage Mission Control",
    description="Space Weather Intelligence Copilot — Dashboard API",
    version="1.0.0",
)

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# State (server-side assessment cache — refreshed on demand)
# ---------------------------------------------------------------------------
_last_assessment: dict[str, Any] = {}


def _get_mode() -> str:
    return os.environ.get("BOBVOYAGE_DATA_PROVIDER", "local").lower()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard HTML."""
    html_path = _STATIC_DIR / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def status():
    """Health check."""
    return {
        "status": "ok",
        "service": "BobVoyage Mission Control",
        "mode": _get_mode(),
    }


@app.get("/api/assessment")
async def assessment(mode: str = Query(default=None)):
    """Run the full BobVoyage intelligence pipeline."""
    global _last_assessment
    effective_mode = mode or ("demo" if _get_mode() == "local" else "live")
    try:
        result = get_assessment(mode=effective_mode)
        _last_assessment = result
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class AssessRequest(BaseModel):
    mission_profile: dict[str, str] | None = None
    mode: str | None = None


@app.post("/api/assess")
async def assess(req: AssessRequest):
    """Re-assess with a custom mission profile."""
    global _last_assessment
    effective_mode = req.mode or ("demo" if _get_mode() == "local" else "live")
    try:
        result = get_assessment(
            mode=effective_mode,
            mission_profile=req.mission_profile,
        )
        _last_assessment = result
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/forecast")
async def forecast(
    param: str = Query(default="solar_wind_speed"),
    mode:  str = Query(default=None),
):
    """Return forecast data for a single parameter."""
    effective_mode = mode or ("demo" if _get_mode() == "local" else "live")
    try:
        result = get_assessment(mode=effective_mode)
        by_param = result.get("forecast_by_param", {})
        conditions = result.get("conditions", {})
        meta = result.get("param_meta", {}).get(param, {"label": param, "unit": ""})
        return JSONResponse(content={
            "param":        param,
            "label":        meta.get("label", param),
            "unit":         meta.get("unit", ""),
            "current_value": conditions.get(param),
            "forecast":     by_param.get(param, []),
            "available_params": list(by_param.keys()),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class AskRequest(BaseModel):
    question: str
    mode: str | None = None


@app.post("/api/ask")
async def ask(req: AskRequest):
    """Route a conversational question through the intelligence pipeline."""
    effective_mode = req.mode or ("demo" if _get_mode() == "local" else "live")
    try:
        # Use cached assessment if available, else fetch fresh
        assessment_data = _last_assessment
        if not assessment_data:
            assessment_data = get_assessment(mode=effective_mode)
        result = ask_question(req.question, assessment_data)
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("BOBVOYAGE_PORT", "8080"))
    mode_label = "LIVE" if _get_mode() != "local" else "DEMO"
    print(f"BobVoyage Mission Control starting — {mode_label} mode — port {port}")
    uvicorn.run("bobvoyage.dashboard.app:app", host="0.0.0.0", port=port, reload=False)
