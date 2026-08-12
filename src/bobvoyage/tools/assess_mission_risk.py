"""
assess_mission_risk — BobVoyage MCP tool

Translates space-weather observations, trends, anomaly-detection results, and
short-term forecasts into an explainable, domain-level spacecraft operational
risk assessment.

===========================================================================
METHODOLOGY
===========================================================================

1. INPUTS
   The tool consumes the structured JSON outputs produced by the four existing
   BobVoyage MCP tools:
     • get_current_conditions  → current observed values
     • analyze_trends          → recent directional changes
     • detect_anomalies        → statistically significant deviations
     • predict_conditions      → short-term forecast

   None of the internal algorithms of those tools are duplicated here.

2. MISSION PROFILE
   The operator (or default) mission profile defines normalised per-domain
   sensitivity on a 3-level scale:
     LOW (1) / MEDIUM (2) / HIGH (3)

   Domains: radiation, communications, navigation, power, attitude_control

3. PARAMETER → DOMAIN MAPPING
   Each space-weather parameter contributes to one or more risk domains via
   a fixed, documented mapping:

   solar_wind_speed     → communications (50%), navigation (30%), attitude (20%)
   solar_wind_density   → communications (40%), attitude (40%), navigation (20%)
   magnetic_field       → navigation (50%), attitude (30%), communications (20%)
   xray_flux            → communications (80%), navigation (20%)
   proton_flux          → radiation (70%), power (20%), electronics (10%)
   geomagnetic_index    → communications (40%), navigation (30%), attitude (30%)

4. ENVIRONMENTAL SEVERITY SCORE  [0, 100]
   For each parameter p in each domain d:

     current_contrib  = normalise(observed_value, p_min, p_max)  × 100
     anomaly_contrib  = z_score_severity(p)   →  0 | 25 | 50
     trend_contrib    = trend_severity(p)     →  0 | 10 | 20 | 35
     forecast_contrib = forecast_change(p)    →  0 | 10 | 20 | 35

     env_score(p) = clip(
         current_contrib × 0.35
       + anomaly_contrib × 0.30
       + trend_contrib   × 0.20
       + forecast_contrib × 0.15
       , 0, 100
     )

5. DOMAIN RISK SCORE  [0, 100]
   domain_score(d) = clip(
       sum_over_p( env_score(p) × weight(p,d) ) × sensitivity_multiplier(d)
       , 0, 100
   )

   sensitivity_multiplier:  LOW=0.6  MEDIUM=1.0  HIGH=1.5

6. OVERALL RISK SCORE  [0, 100]
   overall_score = weighted average of domain scores,
                   weights = sensitivity multipliers

7. RISK LEVEL CLASSIFICATION
   overall_score  < 25  → LOW
   25 ≤ score     < 50  → MODERATE
   50 ≤ score     < 75  → HIGH
   score          ≥ 75  → CRITICAL

   (domain scores use the same thresholds)

8. EVIDENCE TRACEABILITY
   Every driver references its source category:
     OBSERVED / ANALYZED / PREDICTED

9. RECOMMENDATIONS
   Conservative, domain-specific monitoring suggestions are generated
   based on the risk level of each domain.

Responsibility: risk assessment ONLY.
No prediction, anomaly-detection, or data-retrieval logic here.
===========================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default dataset path (used when calling child tools directly)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "space_weather.csv"

# ---------------------------------------------------------------------------
# Mission profile
# ---------------------------------------------------------------------------

SENSITIVITY_LEVELS = {"low", "medium", "high"}
SENSITIVITY_MULTIPLIER = {"low": 0.6, "medium": 1.0, "high": 1.5}
SENSITIVITY_NUMERIC    = {"low": 1,   "medium": 2,   "high": 3}

DEFAULT_MISSION_PROFILE: dict[str, str] = {
    "radiation_sensitivity":       "medium",
    "communications_sensitivity":  "medium",
    "navigation_sensitivity":      "medium",
    "power_sensitivity":           "low",
    "attitude_control_sensitivity":"medium",
}

# ---------------------------------------------------------------------------
# Parameter → domain contribution weights (must sum to 1.0 per parameter)
# ---------------------------------------------------------------------------

_PARAM_DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    "solar_wind_speed": {
        "communications":    0.50,
        "navigation":        0.30,
        "attitude_control":  0.20,
    },
    "solar_wind_density": {
        "communications":    0.40,
        "attitude_control":  0.40,
        "navigation":        0.20,
    },
    "magnetic_field": {
        "navigation":        0.50,
        "attitude_control":  0.30,
        "communications":    0.20,
    },
    "xray_flux": {
        "communications":    0.80,
        "navigation":        0.20,
    },
    "proton_flux": {
        "radiation":         0.70,
        "power":             0.20,
        "attitude_control":  0.10,
    },
    "geomagnetic_index": {
        "communications":    0.40,
        "navigation":        0.30,
        "attitude_control":  0.30,
    },
}

# All risk domains (superset of the profile keys without the _sensitivity suffix)
_ALL_DOMAINS = ["radiation", "communications", "navigation",
                "power", "attitude_control"]

# ---------------------------------------------------------------------------
# Reference ranges for normalising current values to [0, 100]
# These represent the full observable range of each parameter for the
# dev dataset; operators can extend these in production.
# ---------------------------------------------------------------------------
_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "solar_wind_speed":   (200.0,  900.0),   # km/s
    "solar_wind_density": (0.1,    20.0),    # cm⁻³
    "magnetic_field":     (0.1,    30.0),    # nT
    "xray_flux":          (1e-9,   1e-4),    # W/m²  (log-scaled)
    "proton_flux":        (0.001,  1000.0),  # pfu (log-scaled)
    "geomagnetic_index":  (0.0,    9.0),     # Kp index
}

# Parameters that span many orders of magnitude → log-normalised
_LOG_PARAMS = {"xray_flux", "proton_flux"}

# Risk-level thresholds (applied to [0,100] scores)
_RISK_THRESHOLDS = [
    (75.0, "CRITICAL"),
    (50.0, "HIGH"),
    (25.0, "MODERATE"),
    (0.0,  "LOW"),
]

# ---------------------------------------------------------------------------
# Scoring weights (environmental score components)
# ---------------------------------------------------------------------------
_W_CURRENT  = 0.35
_W_ANOMALY  = 0.30
_W_TREND    = 0.20
_W_FORECAST = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_risk(score: float) -> str:
    for threshold, level in _RISK_THRESHOLDS:
        if score >= threshold:
            return level
    return "LOW"


def _normalise(value: float, lo: float, hi: float, log_scale: bool = False) -> float:
    """Map value to [0, 1] within [lo, hi]; optionally log-scaled."""
    if log_scale:
        lo  = math.log10(max(lo, 1e-30))
        hi  = math.log10(max(hi, 1e-30))
        val = math.log10(max(value, 1e-30))
    else:
        val = value
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _anomaly_contribution(param: str, anomalies: list[dict]) -> tuple[float, str | None]:
    """
    Return (score_0_to_50, driver_text | None) for `param` based on anomaly list.
    moderate → 25, significant → 50.
    """
    for a in anomalies:
        if a.get("parameter") == param:
            sev = a.get("severity", "normal")
            z   = a.get("z_score", 0.0) or 0.0
            dir_label = a.get("direction", "")
            direction = "above" if "above" in dir_label else "below"
            text = (
                f"{param.replace('_',' ').title()} "
                f"z={z:+.2f} ({direction} baseline) — "
                f"anomaly severity: {sev}"
            )
            if sev == "significant":
                return 50.0, text
            if sev == "moderate":
                return 25.0, text
    return 0.0, None


def _trend_contribution(param: str, trends: dict) -> tuple[float, str | None]:
    """
    Return (score_0_to_35, driver_text | None) for `param` based on trend dict.
    severity: stable→0, minor→10, moderate→20, significant→35.
    """
    t = trends.get(param)
    if not t:
        return 0.0, None
    sev  = t.get("severity", "stable")
    pct  = t.get("change_percent", 0.0) or 0.0
    dirn = t.get("direction", "stable")
    if sev == "stable":
        return 0.0, None
    scores = {"minor": 10.0, "moderate": 20.0, "significant": 35.0}
    score = scores.get(sev, 0.0)
    text = (
        f"{param.replace('_',' ').title()} trend: "
        f"{dirn} {abs(pct):.1f}% — severity: {sev}"
    )
    return score, text


def _forecast_contribution(
    param: str,
    current_val: float,
    predictions: list[dict],
) -> tuple[float, str | None]:
    """
    Return (score_0_to_35, driver_text | None) for `param` based on forecasts.
    Computes the maximum predicted change relative to the current value.
    """
    param_preds = [p["predicted_value"] for p in predictions
                   if p.get("parameter") == param and p.get("predicted_value") is not None]
    if not param_preds or current_val is None:
        return 0.0, None

    lo, hi = _PARAM_RANGES.get(param, (None, None))
    if lo is None:
        return 0.0, None

    span = hi - lo if hi and lo else 1.0
    if span <= 0:
        return 0.0, None

    max_pred = max(param_preds)
    min_pred = min(param_preds)
    # Max deviation from current in normalised units
    max_deviation = max(abs(max_pred - current_val), abs(min_pred - current_val))
    norm_deviation = max_deviation / span * 100.0  # as percentage of range

    if norm_deviation < 2.0:
        return 0.0, None
    elif norm_deviation < 5.0:
        score = 10.0
    elif norm_deviation < 10.0:
        score = 20.0
    else:
        score = 35.0

    horizon_min = (len(param_preds)) * 5  # assume 5-min sampling
    direction = "upward" if max_pred > current_val else "downward"
    text = (
        f"{param.replace('_',' ').title()} forecast: "
        f"{direction} trend over next ~{horizon_min} min "
        f"(max deviation {norm_deviation:.1f}% of range)"
    )
    return score, text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_mission_risk(
    conditions:   dict | None = None,
    trends:       dict | None = None,
    anomalies:    list | None = None,
    predictions:  list | None = None,
    mission_profile: dict[str, str] | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assess spacecraft operational risk from space-weather intelligence.

    The tool consumes the structured outputs of the four existing BobVoyage
    MCP tools.  If any input is omitted (None), that evidence layer is absent
    from the assessment — the tool adapts gracefully.

    Parameters
    ----------
    conditions:
        Output of ``get_current_conditions()["observation"]``
        (the inner observation dict).
    trends:
        Output of ``analyze_trends()["trends"]``
        (the per-parameter trend dict).
    anomalies:
        Output of ``detect_anomalies()["anomalies"]``
        (the anomaly list).
    predictions:
        Output of ``predict_conditions()["predictions"]``
        (the flat list of per-step, per-parameter forecast dicts).
    mission_profile:
        Dict with keys: radiation_sensitivity, communications_sensitivity,
        navigation_sensitivity, power_sensitivity,
        attitude_control_sensitivity.
        Each value must be 'low', 'medium', or 'high'.
        Defaults to DEFAULT_MISSION_PROFILE.
    dataset_path:
        Unused directly (here for future auto-fetch mode).

    Returns
    -------
    dict with keys:
        status, mission_profile, overall_risk, domains,
        evidence, recommendations, message
    """
    # --- validate / default mission profile ----------------------------------
    profile = dict(DEFAULT_MISSION_PROFILE)
    if mission_profile is not None:
        for key, val in mission_profile.items():
            if not isinstance(val, str) or val.lower() not in SENSITIVITY_LEVELS:
                return _error(
                    f"Invalid sensitivity value '{val}' for '{key}'. "
                    f"Must be one of: {sorted(SENSITIVITY_LEVELS)}."
                )
            profile[key] = val.lower()

    # --- normalise inputs (default to empty) ---------------------------------
    obs:   dict       = conditions  or {}
    trnd:  dict       = trends      or {}
    anoms: list[dict] = anomalies   or []
    preds: list[dict] = predictions or []

    # --- track evidence buckets ----------------------------------------------
    evidence: dict[str, list[str]] = {
        "observed":   [],
        "analyzed":   [],
        "predicted":  [],
    }

    # --- per-parameter environmental scores ----------------------------------
    env_scores: dict[str, float] = {}

    for param, ranges in _PARAM_RANGES.items():
        current_val = obs.get(param)

        # --- current value component
        if current_val is not None:
            try:
                cv = float(current_val)
                log = param in _LOG_PARAMS
                norm = _normalise(cv, ranges[0], ranges[1], log_scale=log)
                current_contrib = norm * 100.0
                evidence["observed"].append(
                    f"{param.replace('_',' ').title()}: "
                    f"{cv:.4g} (normalised {norm*100:.1f}% of reference range)"
                )
            except (TypeError, ValueError):
                current_contrib = 0.0
                cv = None
        else:
            current_contrib = 0.0
            cv = None

        # --- anomaly component
        anomaly_contrib, anomaly_driver = _anomaly_contribution(param, anoms)
        if anomaly_driver:
            evidence["analyzed"].append(f"ANOMALY — {anomaly_driver}")

        # --- trend component
        trend_contrib, trend_driver = _trend_contribution(param, trnd)
        if trend_driver:
            evidence["analyzed"].append(f"TREND — {trend_driver}")

        # --- forecast component
        forecast_contrib, forecast_driver = _forecast_contribution(
            param, cv if cv is not None else 0.0, preds
        )
        if forecast_driver:
            evidence["predicted"].append(f"FORECAST — {forecast_driver}")

        # --- combine
        env_score = (
            current_contrib  * _W_CURRENT  +
            anomaly_contrib  * _W_ANOMALY  +
            trend_contrib    * _W_TREND    +
            forecast_contrib * _W_FORECAST
        )
        env_scores[param] = max(0.0, min(100.0, env_score))

    # --- domain scores -------------------------------------------------------
    profile_key = {
        "radiation":       "radiation_sensitivity",
        "communications":  "communications_sensitivity",
        "navigation":      "navigation_sensitivity",
        "power":           "power_sensitivity",
        "attitude_control":"attitude_control_sensitivity",
    }

    domain_results: list[dict[str, Any]] = []

    for domain in _ALL_DOMAINS:
        sensitivity_key = profile_key[domain]
        sensitivity_str = profile.get(sensitivity_key, "medium")
        multiplier      = SENSITIVITY_MULTIPLIER[sensitivity_str]

        # Weighted sum of environmental scores across contributing parameters
        raw_domain_score = 0.0
        domain_drivers: list[str] = []

        for param, weights in _PARAM_DOMAIN_WEIGHTS.items():
            w = weights.get(domain, 0.0)
            if w == 0.0:
                continue
            e_score = env_scores.get(param, 0.0)
            contribution = e_score * w

            # Collect meaningful drivers (threshold: contributes > 5 pts before multiplier)
            if contribution > 5.0:
                current_val = obs.get(param)
                cv_str = f"{float(current_val):.4g}" if current_val is not None else "unavailable"

                # Anomaly driver
                a_score, a_text = _anomaly_contribution(param, anoms)
                if a_text:
                    domain_drivers.append(f"OBSERVED: {param.replace('_',' ').title()} = {cv_str} — {a_text}")

                # Trend driver
                t_score, t_text = _trend_contribution(param, trnd)
                if t_text:
                    domain_drivers.append(f"ANALYZED: {t_text}")

                # Forecast driver
                f_score, f_text = _forecast_contribution(
                    param, float(current_val) if current_val else 0.0, preds
                )
                if f_text:
                    domain_drivers.append(f"PREDICTED: {f_text}")

                # If no specific driver text but the observed value is elevated,
                # add a plain observed driver
                if not a_text and not t_text and not f_text:
                    domain_drivers.append(
                        f"OBSERVED: {param.replace('_',' ').title()} = {cv_str} "
                        f"(contributes {contribution:.1f} pts to {domain} risk)"
                    )

            raw_domain_score += contribution

        # Apply sensitivity multiplier and clip
        domain_score = min(100.0, raw_domain_score * multiplier)
        risk_level   = _classify_risk(domain_score)

        domain_results.append({
            "domain":      domain,
            "risk":        risk_level,
            "score":       round(domain_score, 1),
            "sensitivity": sensitivity_str,
            "drivers":     list(dict.fromkeys(domain_drivers)),  # deduplicate, preserve order
        })

    # --- overall risk score --------------------------------------------------
    # Weighted average by sensitivity multiplier
    total_weight    = sum(SENSITIVITY_MULTIPLIER[profile[k]] for k in profile_key.values())
    weighted_sum    = sum(
        dr["score"] * SENSITIVITY_MULTIPLIER[profile[profile_key[dr["domain"]]]]
        for dr in domain_results
    )
    overall_score   = min(100.0, weighted_sum / total_weight) if total_weight > 0 else 0.0
    overall_level   = _classify_risk(overall_score)

    # --- recommendations -----------------------------------------------------
    recommendations = _build_recommendations(domain_results, overall_level, profile)

    # --- evidence availability note ------------------------------------------
    missing_layers: list[str] = []
    if not obs:    missing_layers.append("current conditions")
    if not trnd:   missing_layers.append("trend analysis")
    if not anoms and anoms is not None and len(anoms) == 0:
        pass   # empty anomaly list is valid (no anomalies)
    if not preds:  missing_layers.append("forecast")

    data_note = (
        f"Assessment based on available data. "
        f"Missing evidence layers: {missing_layers}." if missing_layers else
        "Assessment based on all four evidence layers: "
        "current conditions, trend analysis, anomaly detection, and forecast."
    )

    return {
        "status":          "ok",
        "mission_profile": profile,
        "overall_risk": {
            "level": overall_level,
            "score": round(overall_score, 1),
        },
        "domains":         domain_results,
        "evidence":        evidence,
        "recommendations": recommendations,
        "message":         data_note,
    }


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

_DOMAIN_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "radiation": {
        "LOW":      "Radiation environment nominal. Routine monitoring sufficient.",
        "MODERATE": "Elevated radiation indicators. Consider deferring radiation-sensitive operations.",
        "HIGH":     "Significant radiation indicators. Review scheduling of radiation-sensitive activities and assess cumulative dose budgets.",
        "CRITICAL": "Extreme radiation indicators. Recommend suspending radiation-sensitive operations pending further assessment.",
    },
    "communications": {
        "LOW":      "RF environment nominal. No communication precautions indicated.",
        "MODERATE": "Elevated geomagnetic or solar activity may affect link margins. Monitor signal quality.",
        "HIGH":     "Communications may be significantly affected. Increase link-margin monitoring and prepare contingency windows.",
        "CRITICAL": "Severe communications disruption risk. Review critical uplink/downlink scheduling.",
    },
    "navigation": {
        "LOW":      "Navigation environment nominal.",
        "MODERATE": "Mild ionospheric disturbance possible. Monitor navigation accuracy.",
        "HIGH":     "Significant navigation degradation possible. Increase navigation solution monitoring.",
        "CRITICAL": "Severe navigation disruption risk. Review operations requiring precise position or timing.",
    },
    "power": {
        "LOW":      "Power environment nominal.",
        "MODERATE": "Mild power-system indicators. Monitor solar-array output.",
        "HIGH":     "Elevated power-system risk. Review operations with high power demand.",
        "CRITICAL": "Severe power-system risk indicators. Reduce non-essential power loads if feasible.",
    },
    "attitude_control": {
        "LOW":      "Attitude and spacecraft operations environment nominal.",
        "MODERATE": "Mild disturbance torque environment. Monitor attitude error.",
        "HIGH":     "Significant attitude disturbance risk. Increase attitude monitoring frequency.",
        "CRITICAL": "Severe attitude disturbance risk. Review manoeuvre plans and consider safe-mode preparedness.",
    },
}

_GENERAL_RECOMMENDATIONS: dict[str, list[str]] = {
    "LOW":      ["Conditions nominal. Maintain standard monitoring cadence."],
    "MODERATE": ["Increase monitoring frequency for affected domains.",
                 "Review mission timeline for sensitive operations."],
    "HIGH":     ["Increase monitoring frequency for all affected domains.",
                 "Review and consider deferring sensitive operations.",
                 "Prepare contingency procedures for further deterioration."],
    "CRITICAL": ["Immediately review all active operations.",
                 "Activate contingency procedures for affected domains.",
                 "Maintain continuous monitoring until conditions improve.",
                 "Consider safe-mode evaluation."],
}


def _build_recommendations(
    domain_results: list[dict],
    overall_level: str,
    profile: dict,
) -> list[str]:
    recs: list[str] = []

    # General recommendation based on overall level
    recs.extend(_GENERAL_RECOMMENDATIONS.get(overall_level, []))

    # Domain-specific recommendations for non-LOW domains
    for dr in domain_results:
        domain = dr["domain"]
        level  = dr["risk"]
        if level != "LOW":
            domain_rec = _DOMAIN_RECOMMENDATIONS.get(domain, {}).get(level)
            if domain_rec:
                recs.append(domain_rec)

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_recs: list[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)

    return unique_recs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _error(message: str) -> dict[str, Any]:
    return {
        "status":          "error",
        "mission_profile": None,
        "overall_risk":    {"level": "UNKNOWN", "score": None},
        "domains":         [],
        "evidence":        {"observed": [], "analyzed": [], "predicted": []},
        "recommendations": [],
        "message":         message,
    }
