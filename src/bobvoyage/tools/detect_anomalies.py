"""
detect_anomalies — BobVoyage MCP tool

Identifies space-weather observations that significantly deviate from a
historical baseline using z-score analysis.

Responsibility: statistical anomaly detection ONLY.
No prediction, trend direction analysis, or risk assessment here.

Design
------
Two non-overlapping windows are drawn from the dataset:

  baseline_window  — older observations that define "normal" behaviour
  recent_window    — the latest observations being examined

For each numeric parameter the baseline produces:
  mean (μ) and standard deviation (σ)

Each value in the recent window is scored:
  z = (value - μ) / σ

|z| < z_threshold           → normal
z_threshold ≤ |z| < z_threshold*1.5 → moderate anomaly
|z| ≥ z_threshold*1.5       → significant anomaly

Only the single worst (highest |z|) observation per parameter is reported
to keep the output actionable.  The full per-observation detail is available
via the `observations` sub-list in each anomaly entry.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "space_weather.csv"

_NUMERIC_PARAMS: list[str] = [
    "solar_wind_speed",
    "solar_wind_density",
    "magnetic_field",
    "xray_flux",
    "proton_flux",
    "geomagnetic_index",
]

# Minimum number of baseline rows required to compute a meaningful std-dev
_MIN_BASELINE_ROWS = 3

# Guard against near-zero denominators
_EPSILON = 1e-30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_severity(abs_z: float, threshold: float) -> str:
    """Classify anomaly severity relative to the detection threshold."""
    if abs_z < threshold:
        return "normal"
    if abs_z < threshold * 1.5:
        return "moderate"
    return "significant"


def _safe_float(value: Any) -> float | None:
    """Convert a value to float; return None on failure or NaN."""
    try:
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, decimals: int = 4) -> float | None:
    """Round for output; preserve full precision for very small values."""
    if value is None:
        return None
    if value != 0 and abs(value) < 1e-3:
        return value
    return round(value, decimals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_anomalies(
    recent_window: int = 6,
    baseline_window: int = 48,
    z_threshold: float = 2.0,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Detect space-weather anomalies using z-score analysis.

    Parameters
    ----------
    recent_window:
        Number of latest observations to examine for anomalies. Must be ≥ 1.
    baseline_window:
        Number of observations immediately before the recent window used to
        establish the baseline (μ, σ).  Must be ≥ 3.
    z_threshold:
        Minimum |z-score| required to flag an observation as anomalous.
        Default 2.0.  Must be > 0.
    dataset_path:
        Optional path override for the CSV.
        Defaults to ``data/space_weather.csv`` at the project root.

    Returns
    -------
    dict with keys:
        status             – "ok" | "error"
        source             – path of the dataset used
        baseline           – {observations, start, end, note}
        analysis_window    – {observations, start, end}
        parameter_stats    – per-parameter {mean, std, skipped_reason}
        anomalies          – list of anomaly objects sorted by |z_score| desc
        summary            – {anomalies_detected, parameters_affected,
                              highest_severity, total_observations_checked}
        message            – human-readable status
    """
    # --- input validation ----------------------------------------------------
    errors: list[str] = []
    if not isinstance(recent_window, int) or recent_window < 1:
        errors.append(f"recent_window must be an integer ≥ 1 (got {recent_window!r}).")
    if not isinstance(baseline_window, int) or baseline_window < _MIN_BASELINE_ROWS:
        errors.append(
            f"baseline_window must be an integer ≥ {_MIN_BASELINE_ROWS} "
            f"(got {baseline_window!r})."
        )
    try:
        z_threshold = float(z_threshold)
        if z_threshold <= 0:
            errors.append(f"z_threshold must be > 0 (got {z_threshold}).")
    except (TypeError, ValueError):
        errors.append(f"z_threshold must be a number (got {z_threshold!r}).")

    if errors:
        return {
            "status": "error",
            "source": None,
            "baseline": None,
            "analysis_window": None,
            "parameter_stats": {},
            "anomalies": [],
            "summary": _empty_summary(),
            "message": "Invalid parameters: " + " ".join(errors),
        }

    path = Path(dataset_path) if dataset_path else _DEFAULT_DATASET

    # --- file validation -----------------------------------------------------
    if not path.exists():
        return _error(str(path), f"Dataset not found at '{path}'.")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return _error(str(path), f"Failed to read dataset: {exc}.")

    if df.empty:
        return _error(str(path), "Dataset is empty — no observations available.")

    # --- sort by timestamp ---------------------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    n_total = len(df)
    needed = recent_window + baseline_window

    # We need at least (baseline_window + recent_window) rows.
    # If the dataset is smaller but still has enough for a baseline, adjust.
    if n_total < _MIN_BASELINE_ROWS + 1:
        return _error(
            str(path),
            f"Insufficient data: dataset has {n_total} valid row(s); "
            f"at least {_MIN_BASELINE_ROWS + 1} are required.",
        )

    # Clamp windows if the dataset is smaller than requested total
    eff_recent = min(recent_window, n_total - _MIN_BASELINE_ROWS)
    eff_baseline = min(baseline_window, n_total - eff_recent)

    baseline_note: str | None = None
    if eff_recent != recent_window or eff_baseline != baseline_window:
        baseline_note = (
            f"Windows adjusted: dataset has only {n_total} rows. "
            f"Used baseline={eff_baseline}, recent={eff_recent}."
        )

    # Split: oldest eff_baseline rows → baseline; last eff_recent rows → recent
    baseline_df = df.iloc[-(eff_baseline + eff_recent) : -eff_recent].reset_index(drop=True)
    recent_df   = df.tail(eff_recent).reset_index(drop=True)

    if baseline_df.empty:
        return _error(
            str(path),
            "Could not construct a baseline window from the available data.",
        )

    # --- window metadata -----------------------------------------------------
    def _ts(sub: pd.DataFrame, pos: int) -> str | None:
        if "timestamp" in sub.columns and len(sub) > 0:
            ts = sub["timestamp"].iloc[pos]
            return ts.isoformat() if pd.notna(ts) else None
        return None

    baseline_meta = {
        "observations": len(baseline_df),
        "start":        _ts(baseline_df, 0),
        "end":          _ts(baseline_df, -1),
        "note":         baseline_note,
    }
    recent_meta = {
        "observations": len(recent_df),
        "start":        _ts(recent_df, 0),
        "end":          _ts(recent_df, -1),
    }

    # --- per-parameter analysis ----------------------------------------------
    parameter_stats: dict[str, Any] = {}
    anomalies: list[dict[str, Any]] = []

    for param in _NUMERIC_PARAMS:
        if param not in df.columns:
            continue

        base_series = pd.to_numeric(baseline_df[param], errors="coerce").dropna()
        if len(base_series) < _MIN_BASELINE_ROWS:
            parameter_stats[param] = {
                "mean": None, "std": None,
                "skipped_reason": (
                    f"Only {len(base_series)} valid baseline values "
                    f"(minimum {_MIN_BASELINE_ROWS} required)."
                ),
            }
            continue

        mu  = float(base_series.mean())
        std = float(base_series.std(ddof=1))  # sample std-dev

        parameter_stats[param] = {
            "mean": _fmt(mu),
            "std":  _fmt(std),
            "skipped_reason": None,
        }

        # Degenerate case: all baseline values are identical (std ≈ 0)
        if std < _EPSILON:
            parameter_stats[param]["skipped_reason"] = (
                "Baseline standard deviation is effectively zero — "
                "z-score undefined; using absolute deviation instead."
            )
            # Fall through to absolute-deviation check below (std treated as special)

        # Score each observation in the recent window
        recent_series = recent_df[param]
        worst: dict[str, Any] | None = None

        for idx, raw in enumerate(recent_series):
            val = _safe_float(raw)
            if val is None:
                continue

            # z-score — or absolute deviation when std ≈ 0
            if std < _EPSILON:
                # When the baseline has no variance, any deviation is anomalous
                z = (val - mu) / _EPSILON if abs(val - mu) > _EPSILON else 0.0
                z = math.copysign(min(abs(z), 999.0), z)  # cap at ±999
            else:
                z = (val - mu) / std

            severity = _classify_severity(abs(z), z_threshold)
            if severity == "normal":
                continue

            direction = "above_baseline" if z > 0 else "below_baseline"

            # Timestamp for this observation
            ts_val: str | None = None
            if "timestamp" in recent_df.columns:
                ts_raw = recent_df["timestamp"].iloc[idx]
                ts_val = ts_raw.isoformat() if pd.notna(ts_raw) else None

            entry = {
                "parameter":     param,
                "timestamp":     ts_val,
                "observed_value": _fmt(val),
                "baseline_mean": _fmt(mu),
                "baseline_std":  _fmt(std),
                "z_score":       _fmt(z, 4),
                "severity":      severity,
                "direction":     direction,
            }

            # Keep only the worst (highest |z|) observation per parameter
            if worst is None or abs(z) > abs(worst["z_score"] or 0):
                worst = entry

        if worst is not None:
            anomalies.append(worst)

    # Sort all anomalies by |z_score| descending
    anomalies.sort(key=lambda a: abs(a["z_score"] or 0), reverse=True)

    # --- summary -------------------------------------------------------------
    severity_rank = {"normal": 0, "moderate": 1, "significant": 2}
    highest = "none"
    if anomalies:
        top = max(anomalies, key=lambda a: severity_rank.get(a["severity"], 0))
        highest = top["severity"]

    total_checked = sum(
        len(pd.to_numeric(recent_df[p], errors="coerce").dropna())
        for p in _NUMERIC_PARAMS
        if p in recent_df.columns
    )

    summary = {
        "anomalies_detected":     len(anomalies) > 0,
        "parameters_affected":    len(anomalies),
        "highest_severity":       highest,
        "total_observations_checked": int(total_checked),
    }

    return {
        "status":           "ok",
        "source":           str(path),
        "baseline":         baseline_meta,
        "analysis_window":  recent_meta,
        "parameter_stats":  parameter_stats,
        "anomalies":        anomalies,
        "summary":          summary,
        "message": (
            f"Anomaly detection complete. "
            f"{len(anomalies)} parameter(s) flagged "
            f"across {len(recent_df)} recent observation(s)."
        ),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _empty_summary() -> dict[str, Any]:
    return {
        "anomalies_detected": False,
        "parameters_affected": 0,
        "highest_severity": "none",
        "total_observations_checked": 0,
    }


def _error(source: str | None, message: str) -> dict[str, Any]:
    return {
        "status":          "error",
        "source":          source,
        "baseline":        None,
        "analysis_window": None,
        "parameter_stats": {},
        "anomalies":       [],
        "summary":         _empty_summary(),
        "message":         message,
    }
