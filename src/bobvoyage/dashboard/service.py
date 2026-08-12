"""
BobVoyage Dashboard Service Layer

Normalises raw BobVoyage tool outputs into clean presentation objects
that the dashboard API and frontend consume.

Contract rules:
  - No analytical logic here.  Every computation is delegated to the
    BobVoyage tools (get_current_conditions, analyze_trends,
    detect_anomalies, predict_conditions, correlate_space_events,
    assess_mission_risk).
  - All fields that a provider cannot supply are represented as None,
    never filled with fabricated values.
  - Causal language prohibited in all user-visible strings.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bobvoyage.data.factory import get_provider
from bobvoyage.data.providers.nasa_donki import NASADonkiProvider
from bobvoyage.tools.current_conditions import get_current_conditions
from bobvoyage.tools.analyze_trends import analyze_trends
from bobvoyage.tools.detect_anomalies import detect_anomalies
from bobvoyage.tools.predict_conditions import predict_conditions
from bobvoyage.tools.correlate_space_events import correlate_space_events
from bobvoyage.tools.assess_mission_risk import assess_mission_risk

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CSV  = _PROJECT_ROOT / "data" / "space_weather.csv"

# ---------------------------------------------------------------------------
# Demo-mode synthetic events (deterministic, no network)
# ---------------------------------------------------------------------------
_DEMO_EVENTS = [
    {
        "event_type":  "CME",
        "event_id":    "DEMO-CME-2025-001",
        "event_time":  "2025-07-20T06:00:00",
        "source":      "NASA_DONKI_DEMO",
        "description": "Moderate CME — partial-halo, LASCO C3",
        "severity":    "moderate",
        "extra":       {},
    },
    {
        "event_type":  "FLR",
        "event_id":    "DEMO-FLR-2025-001",
        "event_time":  "2025-07-20T05:42:00",
        "source":      "NASA_DONKI_DEMO",
        "description": "M2.1 solar flare",
        "severity":    "M2.1",
        "extra":       {},
    },
]

# ---------------------------------------------------------------------------
# Parameter display metadata
# ---------------------------------------------------------------------------
_PARAM_META = {
    "solar_wind_speed":   {"label": "Solar Wind Speed",   "unit": "km/s",   "format": ".0f"},
    "solar_wind_density": {"label": "Solar Wind Density", "unit": "cm⁻³",   "format": ".1f"},
    "magnetic_field":     {"label": "Magnetic Field",     "unit": "nT",     "format": ".1f"},
    "xray_flux":          {"label": "X-ray Flux",         "unit": "W/m²",   "format": ".2e"},
    "proton_flux":        {"label": "Proton Flux",        "unit": "pfu",    "format": ".2f"},
    "geomagnetic_index":  {"label": "Geomagnetic Index",  "unit": "Kp",     "format": ".1f"},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _age_label(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    s = int(seconds)
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m {s % 60}s ago"
    return f"{s // 3600}h {(s % 3600) // 60}m ago"


def _build_timeline(evidence: dict, conditions: dict, risk_result: dict) -> list[dict]:
    """
    Build a chronological intelligence timeline from evidence buckets
    and risk assessment output.  Times are relative labels derived from
    the overall context, since individual tool outputs don't carry
    wall-clock timestamps.
    """
    now = time.time()
    items: list[dict] = []

    for text in evidence.get("observed", [])[:4]:
        items.append({"category": "OBSERVED", "text": text.replace("OBSERVED: ", "")})

    for text in evidence.get("analyzed", [])[:4]:
        clean = text.replace("ANOMALY — ", "").replace("TREND — ", "")
        items.append({"category": "ANALYZED", "text": clean})

    for text in evidence.get("predicted", [])[:3]:
        clean = text.replace("FORECAST — ", "")
        items.append({"category": "PREDICTED", "text": clean})

    for text in evidence.get("correlated", [])[:3]:
        clean = text.replace("CORRELATED — ", "")
        items.append({"category": "CORRELATED", "text": clean})

    # Final assessment entry
    overall = risk_result.get("overall_risk", {})
    level   = overall.get("level", "UNKNOWN")
    score   = overall.get("score", 0)
    items.append({
        "category": "ASSESSMENT",
        "text": f"Mission risk level: {level} ({score}/100)",
    })

    return items


def _build_telemetry(conditions: dict, meta: dict) -> list[dict]:
    """Build normalised telemetry rows for the frontend."""
    rows = []
    for param, pmeta in _PARAM_META.items():
        val = conditions.get(param)
        rows.append({
            "param":     param,
            "label":     pmeta["label"],
            "unit":      pmeta["unit"],
            "value":     val,
            "available": val is not None,
        })
    return rows


def _build_conversational_response(
    question: str,
    assessment: dict,
) -> str:
    """
    Route a natural-language question to the relevant evidence from the
    current assessment.  No separate analytical engine.
    """
    q = question.lower().strip()
    overall = assessment.get("risk", {})
    level   = overall.get("level", "UNKNOWN")
    score   = overall.get("score", 0)

    if any(w in q for w in ("what is happening", "current", "now", "status")):
        obs_lines = assessment.get("evidence", {}).get("observed", [])
        if obs_lines:
            body = "\n".join(f"  • {l}" for l in obs_lines[:5])
            return (
                f"**Current conditions**\n{body}\n\n"
                f"Mission risk is **{level}** ({score}/100)."
            )
        return f"Current mission risk is **{level}** ({score}/100). No observation data available."

    if any(w in q for w in ("why", "reason", "explain", "driver", "high", "cause")):
        domains = assessment.get("risk", {}).get("domains", [])
        high = [d for d in domains if d["risk"] in ("HIGH", "CRITICAL")]
        if not high:
            high = sorted(domains, key=lambda x: -x["score"])[:2]
        lines = []
        for d in high[:2]:
            lines.append(f"**{d['domain'].replace('_',' ').title()}** — {d['risk']} ({d['score']:.0f}/100)")
            for drv in d.get("drivers", [])[:3]:
                lines.append(f"  • {drv}")
        return "\n".join(lines) if lines else "No elevated domain risk detected."

    if any(w in q for w in ("changed", "last hour", "trend", "anomaly")):
        analyzed = assessment.get("evidence", {}).get("analyzed", [])
        if analyzed:
            body = "\n".join(f"  • {l}" for l in analyzed[:5])
            return f"**Recent changes detected:**\n{body}"
        return "No significant trends or anomalies detected in the recent window."

    if any(w in q for w in ("predict", "forecast", "next", "expect", "future")):
        predicted = assessment.get("evidence", {}).get("predicted", [])
        if predicted:
            body = "\n".join(f"  • {l}" for l in predicted[:5])
            return f"**Short-term forecast:**\n{body}"
        return "Forecast data not available in the current assessment."

    if any(w in q for w in ("event", "cme", "flare", "sep", "gst", "storm", "nasa", "external")):
        correlated = assessment.get("evidence", {}).get("correlated", [])
        ces = assessment.get("correlated_events", [])
        if ces:
            lines = []
            for ce in ces[:3]:
                lines.append(
                    f"**{ce['event_type']}** at {ce.get('event_time','?')} — "
                    f"score {ce['correlation_score']:.2f} "
                    f"({ce.get('interpretation','?').replace('_',' ')})"
                )
            return "**Temporally associated space-weather events:**\n" + "\n".join(lines)
        return "No significant space-weather event correlations detected."

    if any(w in q for w in ("recommend", "monitor", "action", "should", "operator")):
        recs = assessment.get("recommendations", [])
        if recs:
            body = "\n".join(f"  • {r}" for r in recs[:5])
            return f"**Recommended actions:**\n{body}"
        return "Conditions nominal. Maintain standard monitoring cadence."

    # Fallback
    return (
        f"**BobVoyage** — Mission risk: **{level}** ({score}/100).\n\n"
        "Try asking:\n"
        "  • What is happening right now?\n"
        "  • Why is communications risk high?\n"
        "  • What changed recently?\n"
        "  • What is predicted?\n"
        "  • Are there any relevant external events?\n"
        "  • What should operators monitor?"
    )


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def get_assessment(
    mode: str = "demo",
    mission_profile: dict | None = None,
    nasa_api_key: str | None = None,
) -> dict[str, Any]:
    """
    Run the full BobVoyage intelligence pipeline and return a normalised
    presentation object for the dashboard.

    Parameters
    ----------
    mode : "demo" | "live"
    mission_profile : optional override for the 5-domain sensitivity dict
    nasa_api_key : NASA API key for live mode
    """
    t_start = time.monotonic()

    # --- select provider -------------------------------------------------------
    if mode == "live":
        obs_provider_name  = "noaa"
        evt_provider_name  = "nasa_donki"
    else:
        obs_provider_name  = "local"
        evt_provider_name  = None   # use demo events

    # --- current conditions ----------------------------------------------------
    obs_provider = get_provider(provider=obs_provider_name)
    conditions_result = get_current_conditions(
        dataset_path=str(_DEFAULT_CSV) if mode == "demo" else None
    )
    conditions = conditions_result.get("observation", {}) or {}
    provider_status = conditions_result.get("status", "ok")
    is_stale        = conditions.get("is_stale", False)
    data_age        = conditions.get("data_age_seconds")
    source_name     = conditions.get("source", obs_provider_name.upper())

    # --- trends ----------------------------------------------------------------
    trends_result = analyze_trends(
        window=12,
        dataset_path=str(_DEFAULT_CSV) if mode == "demo" else None,
    )
    trends = trends_result.get("trends", {}) if trends_result.get("status") == "ok" else {}

    # --- anomalies -------------------------------------------------------------
    anomalies_result = detect_anomalies(
        recent_window=6, baseline_window=48,
        dataset_path=str(_DEFAULT_CSV) if mode == "demo" else None,
    )
    anomalies = (
        anomalies_result.get("anomalies", [])
        if anomalies_result.get("status") == "ok" else []
    )

    # --- forecast --------------------------------------------------------------
    forecast_result = predict_conditions(
        horizon=12, lookback=48,
        dataset_path=str(_DEFAULT_CSV) if mode == "demo" else None,
    )
    predictions = (
        forecast_result.get("predictions", [])
        if forecast_result.get("status") == "ok" else []
    )
    forecast_summary = forecast_result.get("summary", {}) if forecast_result.get("status") == "ok" else {}

    # --- events ----------------------------------------------------------------
    if mode == "live":
        try:
            nasa_prov = NASADonkiProvider(
                api_key=nasa_api_key or os.environ.get("BOBVOYAGE_NASA_API_KEY", "DEMO_KEY")
            )
            evt_resp  = nasa_prov.get_events(days_back=7)
            events    = [e.to_dict() for e in evt_resp.events] if evt_resp.status == "ok" else []
        except Exception:
            events = []
    else:
        events = list(_DEMO_EVENTS)

    # --- correlation -----------------------------------------------------------
    obs_history_result = analyze_trends(
        window=48,
        dataset_path=str(_DEFAULT_CSV) if mode == "demo" else None,
    )
    # Build lightweight observation list from conditions for correlation
    obs_for_corr = [conditions] if conditions else []

    corr_result = correlate_space_events(
        events=events,
        observations=obs_for_corr,
        lookback_hours=4.0,
        lookahead_hours=2.0,
        min_score=0.05,
    )
    correlations = (
        corr_result.get("correlations", [])
        if corr_result.get("status") == "ok" else []
    )

    # --- mission risk ----------------------------------------------------------
    risk_result = assess_mission_risk(
        conditions=conditions,
        trends=trends,
        anomalies=anomalies,
        predictions=predictions,
        correlated_events=correlations,
        mission_profile=mission_profile,
    )

    # --- timeline --------------------------------------------------------------
    timeline = _build_timeline(
        evidence=risk_result.get("evidence", {}),
        conditions=conditions,
        risk_result=risk_result,
    )

    # --- telemetry rows --------------------------------------------------------
    telemetry = _build_telemetry(conditions, _PARAM_META)

    # --- forecast by parameter (for chart) ------------------------------------
    forecast_by_param: dict[str, list[dict]] = {}
    for pred in predictions:
        p = pred.get("parameter", "")
        if p not in forecast_by_param:
            forecast_by_param[p] = []
        forecast_by_param[p].append({
            "step":            pred.get("step"),
            "predicted_value": pred.get("predicted_value"),
            "lower_bound":     pred.get("lower_bound"),
            "upper_bound":     pred.get("upper_bound"),
            "timestamp":       pred.get("timestamp"),
        })

    elapsed = time.monotonic() - t_start

    return {
        "mode":             mode,
        "provider_status":  provider_status,
        "source":           source_name,
        "retrieved_at":     _now_iso(),
        "data_age_seconds": data_age,
        "data_age_label":   _age_label(data_age),
        "is_stale":         is_stale,
        "pipeline_ms":      round(elapsed * 1000, 1),
        "conditions":       conditions,
        "telemetry":        telemetry,
        "trends":           trends,
        "anomalies":        anomalies,
        "forecast_summary": forecast_summary,
        "forecast_by_param": forecast_by_param,
        "forecast_params":  list(forecast_by_param.keys()),
        "events":           events,
        "correlations":     correlations,
        "risk": {
            "level":    risk_result.get("overall_risk", {}).get("level", "UNKNOWN"),
            "score":    risk_result.get("overall_risk", {}).get("score", 0),
            "domains":  risk_result.get("domains", []),
        },
        "correlated_events": risk_result.get("correlated_events", []),
        "evidence":          risk_result.get("evidence", {}),
        "recommendations":   risk_result.get("recommendations", []),
        "mission_profile":   risk_result.get("mission_profile", {}),
        "timeline":          timeline,
        "param_meta":        _PARAM_META,
    }


def ask_question(question: str, current_assessment: dict) -> dict[str, Any]:
    """Route a conversational question to relevant assessment evidence."""
    answer = _build_conversational_response(question, current_assessment)
    return {
        "question": question,
        "answer":   answer,
        "source":   "BobVoyage intelligence pipeline",
    }
