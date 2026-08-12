"""
correlate_space_events — BobVoyage MCP intelligence tool

Identifies temporally associated relationships between:
  • Space-weather telemetry (NOAA observations)
  • Solar/space-weather events (NASA DONKI)

===========================================================================
SCIENTIFIC CONSTRAINT
===========================================================================

This tool establishes TEMPORAL AND STATISTICAL ASSOCIATION only.
It does NOT establish causality.

All generated language is drawn from a controlled vocabulary that
explicitly avoids causal framing.  Allowed interpretation values:

  "no_significant_correlation"
  "weak_temporal_association"
  "moderate_temporal_association"
  "strong_temporal_association"

The word "cause", "caused by", or "due to" is NEVER used.

===========================================================================
METHODOLOGY
===========================================================================

For each NASA DONKI event E:

  1. TEMPORAL WINDOW
     An analysis window is defined around t_event:
       [t_event − lookback_hours, t_event + lookahead_hours]

  2. OBSERVATION FILTERING
     All normalized NOAA observations with a timestamp inside the window
     are selected.

  3. BASELINE DEVIATION (z-score)
     For each in-window observation, the deviation from the mean of the
     observations OUTSIDE the window (or a configured baseline set) is
     computed as a z-score.  This reuses the same z-score formula as
     detect_anomalies without duplicating its code.

  4. TREND ALIGNMENT
     The trend direction (from analyze_trends) within the window is
     scored for severity.

  5. CORRELATION SCORE  [0.0, 1.0]
     A weighted composite of four independent components:

       temporal_score  (inverse-linear decay over the window)  × 0.30
       anomaly_score   (normalised max |z-score| across params) × 0.35
       trend_score     (trend severity mapping)                 × 0.20
       event_weight    (event-type importance weight)           × 0.15

  6. INTERPRETATION
     score < 0.25  → "no_significant_correlation"
     score < 0.50  → "weak_temporal_association"
     score < 0.75  → "moderate_temporal_association"
     score ≥ 0.75  → "strong_temporal_association"

  7. EVIDENCE STRINGS
     Every piece of evidence is prefixed with its epistemic label:
       OBSERVED / ANALYZED / PREDICTED / STATISTICAL

     Causal language is prevented by a _causal_guard() filter.

Responsibility: temporal correlation intelligence ONLY.
No prediction, mission-risk assessment, or data retrieval here.
===========================================================================
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from bobvoyage.data.models.space_weather import (
    SpaceWeatherEvent,
    SpaceWeatherObservation,
)

# ---------------------------------------------------------------------------
# Numeric space-weather parameters available for correlation
# ---------------------------------------------------------------------------
_NUMERIC_PARAMS = [
    "solar_wind_speed",
    "solar_wind_density",
    "magnetic_field",
    "xray_flux",
    "proton_flux",
    "geomagnetic_index",
]

# ---------------------------------------------------------------------------
# Event-type importance weights (not causal weights — temporal/activity proxy)
# ---------------------------------------------------------------------------
_EVENT_WEIGHTS: dict[str, float] = {
    "CME":   1.0,
    "GST":   1.0,
    "SEP":   0.9,
    "FLR":   0.8,
    "ALERT": 0.4,
    "OTHER": 0.2,
}

# Interpretation thresholds
_INTERPRETATION_THRESHOLDS = [
    (0.75, "strong_temporal_association"),
    (0.50, "moderate_temporal_association"),
    (0.25, "weak_temporal_association"),
    (0.00, "no_significant_correlation"),
]

# Trend severity → score
_TREND_SCORES: dict[str, float] = {
    "significant": 1.0,
    "moderate":    0.5,
    "minor":       0.25,
    "stable":      0.0,
}

# Composite weights (must sum to 1.0)
_W_TEMPORAL = 0.30
_W_ANOMALY  = 0.35
_W_TREND    = 0.20
_W_EVENT    = 0.15

# Forbidden causal phrases (lower-case substrings)
_CAUSAL_FORBIDDEN = [
    "caused",
    "due to",
    "resulted in",
    "was triggered by",
    "led to",
    "responsible for",
    "directly caused",
    "impact of the cme",
    "effect of the",
]

# Guard: minimum z-score normalisation denominator
_EPSILON = 1e-30
# z-score normalisation cap (z=5 → anomaly_score=1.0)
_Z_NORM_CAP = 5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_utc(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp to a UTC-aware datetime; return None on failure."""
    if not ts:
        return None
    try:
        s = ts.rstrip("Z")
        if "+" not in s and "T" in s:
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _causal_guard(text: str) -> str:
    """
    Strip or replace forbidden causal phrases from evidence strings.
    Returns the cleaned string.  Raises ValueError if cleaning fails to
    remove all forbidden phrases (defensive programming).
    """
    lower = text.lower()
    for phrase in _CAUSAL_FORBIDDEN:
        if phrase in lower:
            # Replace causal framing with a neutral alternative
            text = text.replace(phrase, "temporally coincident with")
            text = text.replace(phrase.capitalize(), "Temporally coincident with")
            lower = text.lower()
    return text


def _temporal_score(
    obs_dt: datetime,
    event_dt: datetime,
    window_seconds: float,
) -> float:
    """
    Inverse-linear temporal proximity score in [0, 1].
    Score = 1 at t_obs == t_event, decays linearly to 0 at the window boundary.
    """
    diff = abs((obs_dt - event_dt).total_seconds())
    if window_seconds <= 0:
        return 1.0 if diff == 0 else 0.0
    return max(0.0, 1.0 - diff / window_seconds)


def _anomaly_score(z: float | None) -> float:
    """Normalise |z-score| to [0, 1] with cap at _Z_NORM_CAP."""
    if z is None:
        return 0.0
    return min(1.0, abs(z) / _Z_NORM_CAP)


def _interpret(score: float) -> str:
    for threshold, label in _INTERPRETATION_THRESHOLDS:
        if score >= threshold:
            return label
    return "no_significant_correlation"


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _compute_z(value: float, baseline_values: list[float]) -> float | None:
    """Compute z-score of value against a baseline list (sample std-dev)."""
    if len(baseline_values) < 2:
        return None
    mean = sum(baseline_values) / len(baseline_values)
    variance = sum((x - mean) ** 2 for x in baseline_values) / (len(baseline_values) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    denom = std if std > _EPSILON else _EPSILON
    return (value - mean) / denom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correlate_space_events(
    events:            list[dict] | list[SpaceWeatherEvent] | None = None,
    observations:      list[dict] | list[SpaceWeatherObservation] | None = None,
    event_window_hours: float = 6.0,
    lookback_hours:     float = 4.0,
    lookahead_hours:    float = 2.0,
    min_score:          float = 0.1,
) -> dict[str, Any]:
    """
    Correlate space-weather events with telemetry observations.

    Parameters
    ----------
    events:
        List of SpaceWeatherEvent dicts (from NASA DONKI provider) or
        SpaceWeatherEvent dataclass instances.
    observations:
        List of SpaceWeatherObservation dicts (from NOAA provider) or
        SpaceWeatherObservation dataclass instances.
    event_window_hours:
        Total search window in hours, split symmetrically.
        Use lookback_hours / lookahead_hours for asymmetric control.
        If set, overrides both lookback and lookahead with half-value each.
        Default: 6.0 h (3 h before + 3 h after each event).
        Note: lookback_hours / lookahead_hours take precedence when provided.
    lookback_hours:
        Hours before event_time to include in the observation window.
    lookahead_hours:
        Hours after event_time to include in the observation window.
    min_score:
        Minimum composite score to include in the output.  Correlations
        below this threshold are suppressed.  Default 0.1.

    Returns
    -------
    dict with keys:
        status          – "ok" | "error"
        correlations    – list of correlation objects, highest score first
        summary         – {events_analyzed, observations_analyzed,
                           potential_correlations, interpretation_counts}
        parameters      – {lookback_hours, lookahead_hours, min_score}
        message         – human-readable status
    """
    # --- input validation ----------------------------------------------------
    errors: list[str] = []
    if not isinstance(lookback_hours, (int, float)) or lookback_hours <= 0:
        errors.append(f"lookback_hours must be > 0 (got {lookback_hours!r}).")
    if not isinstance(lookahead_hours, (int, float)) or lookahead_hours <= 0:
        errors.append(f"lookahead_hours must be > 0 (got {lookahead_hours!r}).")
    if not isinstance(min_score, (int, float)) or not (0.0 <= min_score <= 1.0):
        errors.append(f"min_score must be in [0, 1] (got {min_score!r}).")
    if errors:
        return _error("Invalid parameters: " + " ".join(errors))

    lookback_s  = lookback_hours  * 3600.0
    lookahead_s = lookahead_hours * 3600.0
    # Window for temporal score decay = larger of the two sides
    window_s    = max(lookback_s, lookahead_s)

    # --- normalise inputs to plain dicts -------------------------------------
    def _to_dict(item: Any) -> dict:
        return item.to_dict() if hasattr(item, "to_dict") else dict(item)

    ev_list: list[dict]  = [_to_dict(e) for e in (events or [])]
    obs_list: list[dict] = [_to_dict(o) for o in (observations or [])]

    if not ev_list:
        return {
            "status":       "ok",
            "correlations": [],
            "summary": {
                "events_analyzed":       0,
                "observations_analyzed": len(obs_list),
                "potential_correlations": 0,
                "interpretation_counts": {},
            },
            "parameters": {
                "lookback_hours":  lookback_hours,
                "lookahead_hours": lookahead_hours,
                "min_score":       min_score,
            },
            "message": "No events provided. Nothing to correlate.",
        }

    if not obs_list:
        return {
            "status":       "ok",
            "correlations": [],
            "summary": {
                "events_analyzed":       len(ev_list),
                "observations_analyzed": 0,
                "potential_correlations": 0,
                "interpretation_counts": {},
            },
            "parameters": {
                "lookback_hours":  lookback_hours,
                "lookahead_hours": lookahead_hours,
                "min_score":       min_score,
            },
            "message": "No observations provided. Nothing to correlate.",
        }

    # --- parse observation timestamps once -----------------------------------
    obs_parsed: list[tuple[datetime | None, dict]] = []
    for o in obs_list:
        dt = _parse_utc(o.get("timestamp"))
        obs_parsed.append((dt, o))

    # Pre-build per-parameter numeric series (for baseline z-score)
    param_series: dict[str, list[float]] = {p: [] for p in _NUMERIC_PARAMS}
    for _, o in obs_parsed:
        for p in _NUMERIC_PARAMS:
            v = _safe_float(o.get(p))
            if v is not None:
                param_series[p].append(v)

    # --- correlate each event ------------------------------------------------
    correlations: list[dict[str, Any]] = []

    for ev in ev_list:
        event_dt = _parse_utc(ev.get("event_time"))
        if event_dt is None:
            continue   # cannot correlate without a timestamp

        event_type = ev.get("event_type", "OTHER").upper()
        event_weight = _EVENT_WEIGHTS.get(event_type, _EVENT_WEIGHTS["OTHER"])

        # Filter observations inside the temporal window
        window_obs: list[tuple[datetime, dict]] = []
        for dt, o in obs_parsed:
            if dt is None:
                continue
            before_ok = (event_dt - dt).total_seconds() <= lookback_s and dt <= event_dt
            after_ok  = (dt - event_dt).total_seconds() <= lookahead_s and dt >= event_dt
            if before_ok or after_ok:
                window_obs.append((dt, o))

        if not window_obs:
            # No observations in window — still record as no correlation
            corr_record = _build_correlation(
                ev=ev, event_dt=event_dt, event_type=event_type,
                event_weight=event_weight,
                window_obs=[],
                param_series=param_series,
                window_s=window_s,
                min_score=min_score,
            )
            if corr_record and corr_record["correlation_score"] >= min_score:
                correlations.append(corr_record)
            continue

        corr_record = _build_correlation(
            ev=ev, event_dt=event_dt, event_type=event_type,
            event_weight=event_weight,
            window_obs=window_obs,
            param_series=param_series,
            window_s=window_s,
            min_score=min_score,
        )
        if corr_record and corr_record["correlation_score"] >= min_score:
            correlations.append(corr_record)

    # Sort by correlation score descending
    correlations.sort(key=lambda c: c["correlation_score"], reverse=True)

    # --- summary -------------------------------------------------------------
    interp_counts: dict[str, int] = {}
    for c in correlations:
        label = c["interpretation"]
        interp_counts[label] = interp_counts.get(label, 0) + 1

    potential = sum(
        1 for c in correlations
        if c["interpretation"] != "no_significant_correlation"
    )

    return {
        "status":       "ok",
        "correlations": correlations,
        "summary": {
            "events_analyzed":       len(ev_list),
            "observations_analyzed": len(obs_list),
            "potential_correlations": potential,
            "interpretation_counts": interp_counts,
        },
        "parameters": {
            "lookback_hours":  lookback_hours,
            "lookahead_hours": lookahead_hours,
            "min_score":       min_score,
        },
        "message": (
            f"Correlation analysis complete. "
            f"{len(ev_list)} event(s) × {len(obs_list)} observation(s) analyzed. "
            f"{potential} potential temporal association(s) identified."
        ),
    }


# ---------------------------------------------------------------------------
# Correlation record builder
# ---------------------------------------------------------------------------

def _build_correlation(
    ev:           dict,
    event_dt:     datetime,
    event_type:   str,
    event_weight: float,
    window_obs:   list[tuple[datetime, dict]],
    param_series: dict[str, list[float]],
    window_s:     float,
    min_score:    float,
) -> dict[str, Any] | None:

    evidence: list[str] = []

    # --- temporal score: closest observation to event -----------------------
    if window_obs:
        min_dist_s = min(abs((dt - event_dt).total_seconds()) for dt, _ in window_obs)
        best_temp_score = _temporal_score(
            event_dt + timedelta(seconds=min_dist_s), event_dt, window_s
        )
    else:
        min_dist_s    = None
        best_temp_score = 0.0

    # --- anomaly score: max |z| across all params in window -----------------
    max_z:         float | None = None
    max_z_param:   str | None   = None
    max_z_obs_ts:  str | None   = None
    max_z_value:   float | None = None

    param_deviations: list[dict[str, Any]] = []

    for param in _NUMERIC_PARAMS:
        baseline = param_series.get(param, [])
        if len(baseline) < 3:
            continue
        for dt, o in window_obs:
            v = _safe_float(o.get(param))
            if v is None:
                continue
            z = _compute_z(v, baseline)
            if z is None:
                continue
            direction = "above" if z > 0 else "below"
            param_deviations.append({
                "parameter":         param,
                "timestamp":         o.get("timestamp"),
                "value":             v,
                "baseline_mean":     round(sum(baseline) / len(baseline), 4),
                "z_score":           round(z, 3),
                "direction":         direction,
            })
            if max_z is None or abs(z) > abs(max_z):
                max_z       = z
                max_z_param = param
                max_z_obs_ts = o.get("timestamp")
                max_z_value  = v

    # Keep only the most significant observation per parameter
    # (highest |z| per param)
    seen_params: set[str] = set()
    top_deviations: list[dict] = []
    for d in sorted(param_deviations, key=lambda x: abs(x["z_score"]), reverse=True):
        if d["parameter"] not in seen_params:
            seen_params.add(d["parameter"])
            top_deviations.append(d)

    a_score = _anomaly_score(max_z)

    # --- trend score: analyse direction change in window --------------------
    trend_sev   = "stable"
    trend_score = 0.0

    if len(window_obs) >= 2:
        # Use first and last observation in the window for each parameter
        sorted_window = sorted(window_obs, key=lambda x: x[0])
        for param in _NUMERIC_PARAMS:
            v_first = _safe_float(sorted_window[0][1].get(param))
            v_last  = _safe_float(sorted_window[-1][1].get(param))
            if v_first is None or v_last is None or v_first == 0:
                continue
            pct = abs((v_last - v_first) / v_first) * 100.0
            if pct >= 30:
                sev = "significant"
            elif pct >= 15:
                sev = "moderate"
            elif pct >= 5:
                sev = "minor"
            else:
                sev = "stable"
            if _TREND_SCORES.get(sev, 0) > trend_score:
                trend_score = _TREND_SCORES[sev]
                trend_sev   = sev

    # --- composite correlation score -----------------------------------------
    composite = (
        best_temp_score * _W_TEMPORAL +
        a_score         * _W_ANOMALY  +
        trend_score     * _W_TREND    +
        event_weight    * _W_EVENT
    )
    composite = round(min(1.0, max(0.0, composite)), 4)

    interpretation = _interpret(composite)

    # --- evidence strings (epistemic labels, no causal language) -------------
    if min_dist_s is not None:
        min_dist_m = round(min_dist_s / 60.0, 1)
        rel = "before" if min_dist_s > 0 and window_obs and \
              min((dt - event_dt).total_seconds() for dt, _ in window_obs) < 0 else "after"
        evidence.append(_causal_guard(
            f"OBSERVED: Closest telemetry reading is {min_dist_m} min "
            f"relative to event time ({event_type})."
        ))

    if max_z is not None and max_z_param:
        direction_str = "above" if max_z > 0 else "below"
        evidence.append(_causal_guard(
            f"STATISTICAL: {max_z_param.replace('_', ' ').title()} "
            f"showed z={max_z:+.2f} ({direction_str} baseline) "
            f"within the temporal window of the {event_type} event."
        ))

    if trend_sev != "stable":
        evidence.append(_causal_guard(
            f"ANALYZED: {trend_sev.title()} parameter variation detected "
            f"within the {event_type} event temporal window."
        ))

    ev_sev = ev.get("severity")
    if ev_sev:
        evidence.append(
            f"OBSERVED: {event_type} event has reported severity: {ev_sev}."
        )

    if interpretation in ("moderate_temporal_association", "strong_temporal_association"):
        evidence.append(
            f"STATISTICAL: Multiple space-weather indicators changed within "
            f"the temporal window. This is consistent with a potentially "
            f"related disturbance — no causal relationship is claimed."
        )

    return {
        "event": {
            "type":        event_type,
            "event_time":  ev.get("event_time"),
            "external_id": ev.get("external_id"),
            "source":      ev.get("source"),
            "severity":    ev.get("severity"),
            "description": (ev.get("description") or "")[:200],
        },
        "observations_in_window":  len(window_obs),
        "temporal_distance_minutes": (
            round(min_dist_s / 60.0, 1) if min_dist_s is not None else None
        ),
        "top_deviations":    top_deviations[:6],   # top 6 parameters
        "correlation_score": composite,
        "interpretation":    interpretation,
        "component_scores": {
            "temporal":    round(best_temp_score, 3),
            "anomaly":     round(a_score, 3),
            "trend":       round(trend_score, 3),
            "event_weight": round(event_weight, 3),
        },
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _error(message: str) -> dict[str, Any]:
    return {
        "status":       "error",
        "correlations": [],
        "summary": {
            "events_analyzed":        0,
            "observations_analyzed":  0,
            "potential_correlations": 0,
            "interpretation_counts":  {},
        },
        "parameters": {},
        "message":    message,
    }
