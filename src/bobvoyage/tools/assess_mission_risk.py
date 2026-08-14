"""
assess_mission_risk — BobVoyage MCP tool

Translates space-weather observations, trends, anomaly-detection results,
short-term forecasts, and (M8) correlated space-weather events into an
explainable, domain-level spacecraft operational risk assessment.

===========================================================================
METHODOLOGY
===========================================================================

1. INPUTS
   The tool consumes the structured JSON outputs produced by the five
   existing BobVoyage MCP tools:
     • get_current_conditions  → current observed values
     • analyze_trends          → recent directional changes
     • detect_anomalies        → statistically significant deviations
     • predict_conditions      → short-term forecast
     • correlate_space_events  → temporal event associations (M8 addition)

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

5. DOMAIN RISK SCORE  [0, 100]  — Phase 1 (environmental)
   raw_domain_score(d) = clip(
       sum_over_p( env_score(p) × weight(p,d) ) × sensitivity_multiplier(d)
       , 0, 100
   )

   sensitivity_multiplier:  LOW=0.6  MEDIUM=1.0  HIGH=1.5

6. EVENT CORRELATION CONTRIBUTION  [0, 25]  — Phase 2 (M8)
   For each correlated event e and domain d:

     base_contrib = correlation_score(e)
                  × event_domain_relevance(e, d)    [see _EVENT_DOMAIN_RELEVANCE]
                  × sensitivity_multiplier(d)
                  × _CORR_SCALE                     [= 40.0]

     env_saturation(d) = min(1.0, raw_domain_score(d) / 50.0)

     discounted_contrib = base_contrib
                        × (1.0 - env_saturation(d) × _OVERLAP_FACTOR)

   All discounted contributions across all events are summed per domain
   and capped at _CORR_CAP (= 25 pts).  This prevents correlation evidence
   from dominating a domain lacking environmental support, and prevents
   double-counting when the underlying physical signal is already captured
   by the environmental score.

   DOUBLE-COUNTING MITIGATION:
   The same physical signal (e.g. a solar-wind speed spike) may appear in:
     • current_contrib (observed value)
     • anomaly_contrib (z-score deviation)
     • trend_contrib   (directional change)
     • forecast_contrib (predicted continuation)
   If a correlated event is driven by the same observation, adding the full
   correlation score would count the same measurement multiple times.
   The env_saturation discount addresses this by scaling the correlation
   contribution down as the domain score rises from direct environmental
   evidence.  The residual weight (1 - _OVERLAP_FACTOR = 0.30) ensures
   the event type and timing still provide contextual intelligence even
   when environmental data is already rich.

   Risk score ≠ failure probability.  This is decision-support, not a
   validated reliability model.

7. DOMAIN FINAL SCORE  [0, 100]
   domain_score(d) = clip(raw_domain_score(d) + corr_addend(d), 0, 100)
   risk_level       = _classify_risk(domain_score(d))

8. OVERALL RISK SCORE  [0, 100]
   overall_score = weighted average of domain_score values,
                   weights = sensitivity multipliers

9. RISK LEVEL CLASSIFICATION
   overall_score  < 25  → LOW
   25 ≤ score     < 50  → MODERATE
   50 ≤ score     < 75  → HIGH
   score          ≥ 75  → CRITICAL

   (domain scores use the same thresholds)

10. EVIDENCE TRACEABILITY
    Every driver references its source category:
      OBSERVED / ANALYZED / PREDICTED / CORRELATED

11. RECOMMENDATIONS
    Conservative, domain-specific monitoring suggestions are generated
    based on the risk level of each domain.

Responsibility: risk assessment ONLY.
No prediction, anomaly-detection, correlation, or data-retrieval logic here.
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
# M8 — Event-to-Domain Relevance Matrix
# ---------------------------------------------------------------------------
# Rows: event type (uppercase).  Columns: domain.
# Values in [0, 1] — represent the degree to which this event type is
# associated with operational risk in each domain.
#
# Rationale (from NOAA/ESA space-weather impacts literature):
#   CME   — drives geomagnetic storms → broad spacecraft impact across all
#            domains; comms/navigation most affected; particle acceleration
#            contributes moderate radiation risk.
#   SEP   — solar energetic particles are the primary radiation threat;
#            high-energy ions also degrade solar-array output (power);
#            secondary ionospheric ionization affects comms moderately.
#   FLR   — solar flare X-ray burst ionizes the ionosphere causing radio
#            blackouts (comms, navigation); brief duration limits power/rad
#            contributions.
#   GST   — geomagnetic storm: disturbs magnetosphere → comms scintillation,
#            severe ionospheric navigation errors, magnetic-torque effects on
#            attitude; induced currents threaten power systems moderately.
#   ALERT — generic NOAA alert; conservative lower weights applied uniformly.
#   OTHER — unknown event type; minimal default weights.
#
# These are decision-support heuristics, not scientifically validated
# probability weights.  They do not represent guaranteed spacecraft effects.
# ---------------------------------------------------------------------------

_EVENT_DOMAIN_RELEVANCE: dict[str, dict[str, float]] = {
    "CME": {
        "radiation":       0.50,
        "communications":  0.90,
        "navigation":      0.80,
        "power":           0.30,
        "attitude_control": 0.60,
    },
    "SEP": {
        "radiation":       0.95,
        "communications":  0.40,
        "navigation":      0.20,
        "power":           0.60,
        "attitude_control": 0.10,
    },
    "FLR": {
        "radiation":       0.20,
        "communications":  0.90,
        "navigation":      0.60,
        "power":           0.10,
        "attitude_control": 0.05,
    },
    "GST": {
        "radiation":       0.40,
        "communications":  0.90,
        "navigation":      0.85,
        "power":           0.20,
        "attitude_control": 0.70,
    },
    "ALERT": {
        "radiation":       0.10,
        "communications":  0.30,
        "navigation":      0.20,
        "power":           0.10,
        "attitude_control": 0.05,
    },
    "OTHER": {
        "radiation":       0.05,
        "communications":  0.10,
        "navigation":      0.05,
        "power":           0.05,
        "attitude_control": 0.05,
    },
}

# ---------------------------------------------------------------------------
# M8 — Correlation contribution parameters
# ---------------------------------------------------------------------------
# Scale factor: max pts a single event can contribute before env-saturation
# discount and cap.  Kept below 50 so a single perfect-score event without
# environmental support cannot independently push a domain to HIGH.
_CORR_SCALE = 40.0

# Overlap discount factor.  At full env-saturation (env_score ≥ 50 pts)
# the correlation contribution is reduced to (1 - 0.70) = 30% of its base
# value.  Retains contextual intelligence while preventing double-counting.
_OVERLAP_FACTOR = 0.70

# Hard cap on the total correlation addend per domain across all events.
# Prevents a pile of weak events from fabricating HIGH risk without
# direct environmental evidence.
_CORR_CAP = 25.0

# Causal-language prevention: forbidden substrings in evidence strings.
_CAUSAL_FORBIDDEN_SUBSTRINGS = [
    "caused",
    "due to",
    "resulted in",
    "was triggered by",
    "led to",
    "responsible for",
    "directly caused",
]


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


def _causal_guard_risk(text: str) -> str:
    """Strip forbidden causal phrases from risk-assessment evidence strings."""
    lower = text.lower()
    for phrase in _CAUSAL_FORBIDDEN_SUBSTRINGS:
        if phrase in lower:
            text = text.replace(phrase, "temporally associated with")
            text = text.replace(phrase.capitalize(), "Temporally associated with")
            lower = text.lower()
    return text


def _get_event_domain_relevance(event_type: str, domain: str) -> float:
    """Return domain relevance for the given event type (case-insensitive)."""
    et = str(event_type).upper()
    row = _EVENT_DOMAIN_RELEVANCE.get(et, _EVENT_DOMAIN_RELEVANCE["OTHER"])
    return row.get(domain, 0.0)


def _compute_corr_addend(
    correlations: list[dict],
    domain: str,
    raw_domain_score: float,
    multiplier: float,
) -> tuple[float, list[dict]]:
    """
    Compute the total correlation contribution addend for a domain, applying
    env-saturation discount and hard cap.

    Parameters
    ----------
    correlations:
        List of correlation objects from correlate_space_events().
    domain:
        Target risk domain name.
    raw_domain_score:
        Pre-correlation domain score (0–100); used for env-saturation.
    multiplier:
        Mission sensitivity multiplier for this domain.

    Returns
    -------
    (addend, per_event_contributions)
    addend: float in [0, _CORR_CAP]
    per_event_contributions: list of dicts with per-event breakdown
    """
    env_saturation = min(1.0, raw_domain_score / 50.0)
    discount       = 1.0 - env_saturation * _OVERLAP_FACTOR

    total = 0.0
    per_event: list[dict] = []

    for corr in correlations:
        score = corr.get("correlation_score", 0.0) or 0.0
        etype = str(corr.get("event_type", "OTHER")).upper()
        relevance = _get_event_domain_relevance(etype, domain)

        if relevance == 0.0:
            continue

        base = score * relevance * multiplier * _CORR_SCALE
        contrib = base * discount

        if contrib > 0.0:
            per_event.append({
                "event_type":    etype,
                "event_id":      corr.get("event_id"),
                "relevance":     round(relevance, 3),
                "base_contrib":  round(base, 2),
                "discounted":    round(contrib, 2),
            })
            total += contrib

    # Hard cap
    total = min(total, _CORR_CAP)
    return round(total, 2), per_event


def _build_correlated_events_output(
    correlations: list[dict],
    domain_results_raw: dict[str, float],
    profile: dict,
    profile_key: dict[str, str],
) -> list[dict]:
    """
    Build the correlated_events section of the output.

    For each correlation, reports:
    - event metadata
    - which domains are affected (relevance > 0)
    - per-domain risk_contribution (the discounted addend, pre-cap)
    - guarded evidence strings

    Returns a list sorted by correlation_score descending.
    """
    result: list[dict] = []

    for corr in correlations:
        score  = corr.get("correlation_score", 0.0) or 0.0
        ev_sub = corr.get("event") or {}
        etype  = str(ev_sub.get("type") or corr.get("event_type") or "OTHER").upper()

        # Per-domain contributions (pre-cap, for reporting)
        affected_domains: list[str] = []
        risk_contribution: dict[str, float] = {}

        for domain in _ALL_DOMAINS:
            relevance   = _get_event_domain_relevance(etype, domain)
            if relevance == 0.0:
                continue
            sens_key    = profile_key[domain]
            multiplier  = SENSITIVITY_MULTIPLIER[profile.get(sens_key, "medium")]
            raw_score   = domain_results_raw.get(domain, 0.0)
            env_sat     = min(1.0, raw_score / 50.0)
            discount    = 1.0 - env_sat * _OVERLAP_FACTOR
            base        = score * relevance * multiplier * _CORR_SCALE
            discounted  = base * discount

            if discounted >= 0.5:   # report only meaningful contributions
                affected_domains.append(domain)
                risk_contribution[domain] = round(discounted, 1)

        # Build guarded evidence strings
        raw_evidence = corr.get("evidence", [])
        if not raw_evidence:
            raw_evidence = [
                f"{etype} event (id: {corr.get('event_id', 'unknown')}) "
                f"temporally associated with observed measurements in the analysis window."
            ]
        guarded_evidence = [_causal_guard_risk(e) for e in raw_evidence]

        # Add interpretation-level evidence statement
        interp = corr.get("interpretation", "")
        interp_text = (
            f"{etype} correlation score {score:.2f}: {interp.replace('_', ' ')} "
            f"with observed space-weather measurements."
        )
        guarded_evidence.insert(0, _causal_guard_risk(interp_text))

        result.append({
            "event_type":         etype,
            "event_id":           ev_sub.get("external_id") or corr.get("event_id"),
            "event_time":         ev_sub.get("event_time") or corr.get("event_time"),
            "correlation_score":  round(score, 3),
            "interpretation":     interp,
            "affected_domains":   affected_domains,
            "risk_contribution":  risk_contribution,
            "observations_in_window": corr.get("observations_in_window", 0),
            "evidence":           guarded_evidence,
        })

    # Sort by correlation score descending (already sorted from correlate_space_events)
    result.sort(key=lambda x: x["correlation_score"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_mission_risk(
    conditions:        dict | None = None,
    trends:            dict | None = None,
    anomalies:         list | None = None,
    predictions:       list | None = None,
    correlated_events: list | None = None,
    mission_profile:   dict[str, str] | None = None,
    dataset_path:      str | Path | None = None,
) -> dict[str, Any]:
    """Assess spacecraft operational risk from space-weather intelligence.

    The tool consumes the structured outputs of the BobVoyage MCP tools.
    If any input is omitted (None), that evidence layer is absent from the
    assessment — the tool adapts gracefully.

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
    correlated_events:
        Output of ``correlate_space_events()["correlations"]``
        (the correlation list).  M8 addition.  When supplied, each
        correlated event contributes a discounted, env-saturation-adjusted
        risk addend to relevant domains.
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
        correlated_events, evidence, recommendations, message
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
    obs:   dict       = conditions         or {}
    trnd:  dict       = trends             or {}
    anoms: list[dict] = anomalies          or []
    preds: list[dict] = predictions        or []
    corrs: list[dict] = correlated_events  or []

    # --- track evidence buckets ----------------------------------------------
    evidence: dict[str, list[str]] = {
        "observed":    [],
        "analyzed":    [],
        "predicted":   [],
        "correlated":  [],
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

    # --- profile lookup helpers ----------------------------------------------
    profile_key: dict[str, str] = {
        "radiation":       "radiation_sensitivity",
        "communications":  "communications_sensitivity",
        "navigation":      "navigation_sensitivity",
        "power":           "power_sensitivity",
        "attitude_control":"attitude_control_sensitivity",
    }

    # =========================================================================
    # Phase 1: Raw domain scores (environmental evidence only)
    # =========================================================================
    raw_domain_scores: dict[str, float] = {}

    for domain in _ALL_DOMAINS:
        sensitivity_key = profile_key[domain]
        sensitivity_str = profile.get(sensitivity_key, "medium")
        multiplier      = SENSITIVITY_MULTIPLIER[sensitivity_str]

        raw_score = 0.0
        for param, weights in _PARAM_DOMAIN_WEIGHTS.items():
            w       = weights.get(domain, 0.0)
            e_score = env_scores.get(param, 0.0)
            raw_score += e_score * w

        raw_domain_scores[domain] = min(100.0, raw_score * multiplier)

    # =========================================================================
    # Phase 2: Correlation addends (M8) — env-saturation discounted
    # =========================================================================
    corr_addends: dict[str, float] = {d: 0.0 for d in _ALL_DOMAINS}
    corr_per_domain_detail: dict[str, list[dict]] = {d: [] for d in _ALL_DOMAINS}

    if corrs:
        for domain in _ALL_DOMAINS:
            sensitivity_str = profile.get(profile_key[domain], "medium")
            multiplier      = SENSITIVITY_MULTIPLIER[sensitivity_str]
            addend, details = _compute_corr_addend(
                correlations=corrs,
                domain=domain,
                raw_domain_score=raw_domain_scores[domain],
                multiplier=multiplier,
            )
            corr_addends[domain] = addend
            corr_per_domain_detail[domain] = details

        # Populate correlated evidence bucket
        for corr in corrs:
            score = corr.get("correlation_score", 0.0) or 0.0
            etype = str(corr.get("event_type", "OTHER")).upper()
            interp = corr.get("interpretation", "no_significant_correlation")
            ev_time = corr.get("event_time", "unknown time")

            text = (
                f"CORRELATED — {etype} event at {ev_time}: "
                f"score {score:.2f} ({interp.replace('_', ' ')}). "
                f"Temporally associated with the analysis window."
            )
            evidence["correlated"].append(_causal_guard_risk(text))

    # =========================================================================
    # Phase 3: Final domain scores = raw + corr addend
    # =========================================================================
    domain_results: list[dict[str, Any]] = []

    for domain in _ALL_DOMAINS:
        sensitivity_key = profile_key[domain]
        sensitivity_str = profile.get(sensitivity_key, "medium")
        multiplier      = SENSITIVITY_MULTIPLIER[sensitivity_str]

        # Rebuild domain drivers (same logic as before, using env_scores)
        domain_drivers: list[str] = []
        for param, weights in _PARAM_DOMAIN_WEIGHTS.items():
            w = weights.get(domain, 0.0)
            if w == 0.0:
                continue
            e_score = env_scores.get(param, 0.0)
            contribution = e_score * w

            if contribution > 5.0:
                current_val = obs.get(param)
                cv_str = f"{float(current_val):.4g}" if current_val is not None else "unavailable"

                a_score, a_text = _anomaly_contribution(param, anoms)
                if a_text:
                    domain_drivers.append(f"OBSERVED: {param.replace('_',' ').title()} = {cv_str} — {a_text}")

                t_score, t_text = _trend_contribution(param, trnd)
                if t_text:
                    domain_drivers.append(f"ANALYZED: {t_text}")

                f_score, f_text = _forecast_contribution(
                    param, float(current_val) if current_val else 0.0, preds
                )
                if f_text:
                    domain_drivers.append(f"PREDICTED: {f_text}")

                if not a_text and not t_text and not f_text:
                    domain_drivers.append(
                        f"OBSERVED: {param.replace('_',' ').title()} = {cv_str} "
                        f"(contributes {contribution:.1f} pts to {domain} risk)"
                    )

        # Add correlation drivers for this domain
        for detail in corr_per_domain_detail.get(domain, []):
            if detail["discounted"] >= 0.5:
                etype  = detail["event_type"]
                contrib = detail["discounted"]
                domain_drivers.append(
                    f"CORRELATED: {etype} event (id: {detail['event_id']}) — "
                    f"domain relevance {detail['relevance']:.2f}, "
                    f"contribution +{contrib:.1f} pts to {domain} risk "
                    f"(temporally associated with observed conditions)"
                )

        # Final domain score
        domain_score = min(100.0, raw_domain_scores[domain] + corr_addends[domain])
        risk_level   = _classify_risk(domain_score)

        domain_results.append({
            "domain":              domain,
            "risk":                risk_level,
            "score":               round(domain_score, 1),
            "score_environmental": round(raw_domain_scores[domain], 1),
            "score_correlation":   round(corr_addends[domain], 1),
            "sensitivity":         sensitivity_str,
            "drivers":             list(dict.fromkeys(domain_drivers)),
        })

    # --- overall risk score --------------------------------------------------
    total_weight    = sum(SENSITIVITY_MULTIPLIER[profile[k]] for k in profile_key.values())
    weighted_sum    = sum(
        dr["score"] * SENSITIVITY_MULTIPLIER[profile[profile_key[dr["domain"]]]]
        for dr in domain_results
    )
    overall_score   = min(100.0, weighted_sum / total_weight) if total_weight > 0 else 0.0
    overall_level   = _classify_risk(overall_score)

    # --- correlated events output section ------------------------------------
    corr_events_output = _build_correlated_events_output(
        correlations=corrs,
        domain_results_raw=raw_domain_scores,
        profile=profile,
        profile_key=profile_key,
    )

    # --- recommendations -----------------------------------------------------
    recommendations = _build_recommendations(domain_results, overall_level, profile)

    # --- evidence availability note ------------------------------------------
    missing_layers: list[str] = []
    if not obs:    missing_layers.append("current conditions")
    if not trnd:   missing_layers.append("trend analysis")
    if not preds:  missing_layers.append("forecast")
    if not corrs:  missing_layers.append("event correlation")

    layers_present: list[str] = []
    if obs:    layers_present.append("current conditions")
    if trnd:   layers_present.append("trend analysis")
    if anoms is not None:  layers_present.append("anomaly detection")
    if preds:  layers_present.append("forecast")
    if corrs:  layers_present.append("event correlation")

    if missing_layers:
        data_note = (
            f"Assessment based on available data. "
            f"Missing evidence layers: {missing_layers}."
        )
    else:
        data_note = (
            "Assessment based on all five evidence layers: "
            "current conditions, trend analysis, anomaly detection, "
            "forecast, and event correlation."
        )

    return {
        "status":           "ok",
        "mission_profile":  profile,
        "overall_risk": {
            "level": overall_level,
            "score": round(overall_score, 1),
        },
        "domains":          domain_results,
        "correlated_events": corr_events_output,
        "evidence":          evidence,
        "recommendations":   recommendations,
        "message":           data_note,
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
        "status":            "error",
        "mission_profile":   None,
        "overall_risk":      {"level": "UNKNOWN", "score": None},
        "domains":           [],
        "correlated_events": [],
        "evidence":          {"observed": [], "analyzed": [], "predicted": [], "correlated": []},
        "recommendations":   [],
        "message":           message,
    }
