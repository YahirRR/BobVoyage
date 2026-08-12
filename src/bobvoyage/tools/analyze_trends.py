"""
analyze_trends — BobVoyage MCP tool

Analyzes recent space-weather observations and identifies meaningful trends.

Responsibility: descriptive trend analysis ONLY.
No prediction, anomaly detection, or risk assessment is performed here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset location — mirrors the pattern from current_conditions.py
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "space_weather.csv"

# Parameters that will be trend-analyzed when present in the dataset
_NUMERIC_PARAMS: list[str] = [
    "solar_wind_speed",
    "solar_wind_density",
    "magnetic_field",
    "xray_flux",
    "proton_flux",
    "geomagnetic_index",
]

# Thresholds for severity classification (absolute % change)
_SEVERITY_THRESHOLDS = {
    "low":      5.0,   # < 5 %  → stable
    "minor":   15.0,   # 5–15 % → minor
    "moderate": 30.0,  # 15–30 % → moderate
    # ≥ 30 %           → significant
}

# Minimum absolute value used as denominator guard to avoid division by ~0
_EPSILON = 1e-12


def _classify_direction(change_pct: float, stable_threshold: float = 5.0) -> str:
    """Return 'increasing', 'decreasing', or 'stable' based on % change."""
    if abs(change_pct) < stable_threshold:
        return "stable"
    return "increasing" if change_pct > 0 else "decreasing"


def _classify_severity(abs_change_pct: float) -> str:
    if abs_change_pct < _SEVERITY_THRESHOLDS["low"]:
        return "stable"
    if abs_change_pct < _SEVERITY_THRESHOLDS["minor"]:
        return "minor"
    if abs_change_pct < _SEVERITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "significant"


def _safe_round(value: float, decimals: int = 4) -> float | None:
    """Round a float; return None if the value is NaN or infinite.

    For very small values (|x| < 1e-3) the full float precision is preserved
    so that scientific-notation fields like xray_flux are not truncated to 0.0.
    """
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    if value != 0 and abs(value) < 1e-3:
        return value  # preserve full precision for scientific-notation quantities
    return round(value, decimals)


def analyze_trends(
    window: int = 12,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze recent space-weather observations and return trend information.

    Parameters
    ----------
    window:
        Number of most-recent observations to include in the analysis.
        Must be ≥ 2.  Default is 12 (≈ 1 hour at 5-minute resolution).
    dataset_path:
        Optional path override for the CSV dataset.
        Defaults to ``data/space_weather.csv`` at the project root.

    Returns
    -------
    dict with keys:
        status            – "ok" | "error"
        source            – absolute path of the dataset used
        window            – {observations, start, end}
        trends            – per-parameter trend objects
        significant_trends – list of trends whose severity is not "stable",
                             sorted by |change_pct| descending
        message           – human-readable status message
    """
    # --- validate window ------------------------------------------------------
    if not isinstance(window, int) or window < 2:
        return {
            "status": "error",
            "source": None,
            "window": None,
            "trends": {},
            "significant_trends": [],
            "message": f"Invalid window '{window}'. Must be an integer ≥ 2.",
        }

    path = Path(dataset_path) if dataset_path else _DEFAULT_DATASET

    # --- validate file exists -------------------------------------------------
    if not path.exists():
        return {
            "status": "error",
            "source": str(path),
            "window": None,
            "trends": {},
            "significant_trends": [],
            "message": f"Dataset not found at '{path}'.",
        }

    # --- load CSV -------------------------------------------------------------
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "source": str(path),
            "window": None,
            "trends": {},
            "significant_trends": [],
            "message": f"Failed to read dataset: {exc}",
        }

    if df.empty:
        return {
            "status": "error",
            "source": str(path),
            "window": None,
            "trends": {},
            "significant_trends": [],
            "message": "Dataset is empty — no observations available.",
        }

    # --- sort by timestamp (when present) ------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # --- check we have enough rows -------------------------------------------
    n_total = len(df)
    if n_total < 2:
        return {
            "status": "error",
            "source": str(path),
            "window": None,
            "trends": {},
            "significant_trends": [],
            "message": (
                f"Insufficient data: dataset contains {n_total} valid row(s); "
                "at least 2 are required for trend analysis."
            ),
        }

    # Clamp window to available rows (no error — just use what we have)
    effective_window = min(window, n_total)
    subset = df.tail(effective_window).reset_index(drop=True)

    # --- window metadata ------------------------------------------------------
    if "timestamp" in subset.columns:
        ts_start = subset["timestamp"].iloc[0]
        ts_end   = subset["timestamp"].iloc[-1]
        window_start = ts_start.isoformat() if pd.notna(ts_start) else None
        window_end   = ts_end.isoformat()   if pd.notna(ts_end)   else None
    else:
        window_start = window_end = None

    # --- compute per-parameter trends ----------------------------------------
    trends: dict[str, Any] = {}
    significant_trends: list[dict[str, Any]] = []

    for param in _NUMERIC_PARAMS:
        if param not in subset.columns:
            continue

        series = pd.to_numeric(subset[param], errors="coerce").dropna()
        if len(series) < 2:
            continue

        start_val = float(series.iloc[0])
        end_val   = float(series.iloc[-1])

        # Percentage change — guard against near-zero denominators
        denom = abs(start_val) if abs(start_val) > _EPSILON else _EPSILON
        change_pct = ((end_val - start_val) / denom) * 100.0

        abs_change    = end_val - start_val
        direction     = _classify_direction(change_pct)
        severity      = _classify_severity(abs(change_pct))

        trends[param] = {
            "direction":      direction,
            "change_percent": _safe_round(change_pct, 2),
            "change_absolute": _safe_round(abs_change, 4),
            "start_value":    _safe_round(start_val, 4),
            "end_value":      _safe_round(end_val, 4),
            "severity":       severity,
            "observations_used": int(len(series)),
        }

        if severity != "stable":
            significant_trends.append({
                "parameter": param,
                "direction": direction,
                "change_percent": _safe_round(change_pct, 2),
                "severity": severity,
            })

    # Sort significant trends by magnitude of change, largest first
    significant_trends.sort(key=lambda x: abs(x["change_percent"] or 0), reverse=True)

    return {
        "status": "ok",
        "source": str(path),
        "window": {
            "observations": effective_window,
            "start": window_start,
            "end":   window_end,
        },
        "trends": trends,
        "significant_trends": significant_trends,
        "message": (
            f"Trend analysis complete over {effective_window} observations."
        ),
    }
