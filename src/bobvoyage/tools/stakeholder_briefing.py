"""
stakeholder_briefing — BobVoyage MCP tool

Translates the structured output of assess_mission_risk() into clear,
audience-specific natural language, without introducing information that
is not present in the upstream risk assessment.

===========================================================================
SCIENTIFIC CONSTRAINT
===========================================================================

This tool performs TRANSLATION AND FILTERING only.
It does not compute, infer, or derive any new risk values.
All risk levels (LOW / MODERATE / HIGH / CRITICAL) and all driver text
are sourced verbatim or paraphrased directly from the risk_assessment
input; no escalation or downgrading of risk is performed here.

No causal language is introduced.  Action items are framed as
precautionary recommendations, consistent with the epistemic posture of
the rest of the BobVoyage pipeline.

===========================================================================
METHODOLOGY
===========================================================================

1. AUDIENCE VALIDATION
   The `audience` parameter must be one of four supported values:
     "satellite_operator" / "astronaut" / "aviation" / "power_grid"
   Unsupported audiences return status="error".

2. DOMAIN FILTERING
   Each audience maps to a subset of the five risk domains defined in
   assess_mission_risk (radiation, communications, navigation, power,
   attitude_control):

     satellite_operator → communications, navigation, attitude_control, power
     astronaut          → radiation (dominant), communications
     aviation           → communications, navigation
     power_grid         → power, communications
                          (communications proxies geomagnetic disturbance,
                           since assess_mission_risk has no dedicated "grid"
                           domain; the geomagnetic_index parameter feeds the
                           communications domain score, making it the best
                           available proxy for geomagnetically induced
                           currents)

3. RISK SUMMARY
   A 1-2 sentence plain-language summary is generated from
   overall_risk["level"] and overall_risk["score"], translated to the
   vocabulary of the target audience.  No new numeric threshold is
   introduced — the exact level string from assess_mission_risk is
   preserved in the output.

4. RELEVANT DOMAINS
   For each domain in the audience's domain list, the tool extracts the
   risk level, score, and the top-3 most significant driver strings.
   Driver strings are simplified: epistemic prefixes (OBSERVED: /
   ANALYZED: / PREDICTED: / CORRELATED:) are preserved as-is to
   maintain traceability, but technical parameter names are translated
   to plain English using the _PARAM_PLAIN_NAMES map.

5. ACTION ITEMS
   The recommendations list from the risk_assessment is filtered to keep
   only entries that are relevant to the audience's domain set.
   If no domain-specific recommendations match, the general
   recommendations (not domain-filtered) are retained.

6. EVIDENCE NOTE
   A single sentence summarising which evidence layers were present in
   the upstream risk_assessment (derived from the evidence keys whose
   lists are non-empty).

Responsibility: audience translation ONLY.
No risk computation, sensor data access, or event correlation here.
===========================================================================
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Supported audiences and their relevant domains
# ---------------------------------------------------------------------------

_AUDIENCE_DOMAINS: dict[str, list[str]] = {
    "satellite_operator": ["communications", "navigation", "attitude_control", "power"],
    "astronaut":          ["radiation", "communications"],
    "aviation":           ["communications", "navigation"],
    "power_grid":         ["power", "communications"],
}

_SUPPORTED_AUDIENCES: frozenset[str] = frozenset(_AUDIENCE_DOMAINS)

# ---------------------------------------------------------------------------
# Audience-specific vocabulary for the risk summary sentence
# ---------------------------------------------------------------------------

_AUDIENCE_CONTEXT: dict[str, dict[str, str]] = {
    "satellite_operator": {
        "LOW":      "Space-weather conditions are currently nominal. No operational precautions are indicated for satellite operations.",
        "MODERATE": "Space-weather conditions are moderately elevated. Satellite operations should maintain heightened monitoring of link margins, navigation accuracy, and attitude performance.",
        "HIGH":     "Significant space-weather activity is present. Satellite operators should review scheduling of sensitive operations and prepare contingency procedures.",
        "CRITICAL": "Severe space-weather conditions are indicated. Satellite operators should immediately review all active operations and activate contingency procedures.",
    },
    "astronaut": {
        "LOW":      "Radiation and communication conditions are currently nominal. No precautionary action is required.",
        "MODERATE": "Elevated space-weather indicators may increase radiation exposure and affect communication quality. Crew should be aware and monitor conditions.",
        "HIGH":     "Significant radiation and communication risk indicators are present. Review EVA schedules and crew radiation exposure budgets. Ensure communication contingency channels are available.",
        "CRITICAL": "Severe radiation and communication risk indicators are present. EVA activities should be suspended pending further assessment. Continuous monitoring is required.",
    },
    "aviation": {
        "LOW":      "HF communication and GPS navigation conditions are currently nominal. Standard flight operations are not affected.",
        "MODERATE": "Mildly elevated ionospheric activity may degrade HF radio communications and GPS accuracy on some routes. Crews should monitor communication and navigation performance.",
        "HIGH":     "Significant ionospheric disturbance is indicated, which may substantially degrade HF communications and GPS navigation. Consider alternative communication channels and navigation back-ups.",
        "CRITICAL": "Severe ionospheric conditions are indicated. HF communications and GPS navigation may experience significant outages. Review flight plans and activate contingency procedures.",
    },
    "power_grid": {
        "LOW":      "Geomagnetic activity is currently nominal. No elevated risk of geomagnetically induced currents (GIC) is indicated.",
        "MODERATE": "Mildly elevated geomagnetic activity may produce minor geomagnetically induced currents on long transmission lines. Standard monitoring is advised.",
        "HIGH":     "Elevated geomagnetic disturbance indicators may produce significant GIC effects. Grid operators should increase transformer monitoring and review protection settings.",
        "CRITICAL": "Severe geomagnetic disturbance is indicated. The risk of significant geomagnetically induced currents is elevated. Consider activating GIC protection procedures and coordinating with network operators.",
    },
}

# ---------------------------------------------------------------------------
# Domain labels translated to plain English per audience
# ---------------------------------------------------------------------------

_DOMAIN_PLAIN_NAMES: dict[str, dict[str, str]] = {
    "satellite_operator": {
        "communications":  "RF / telemetry link quality",
        "navigation":      "GPS / navigation accuracy",
        "attitude_control":"attitude and orbit control",
        "power":           "power system / solar arrays",
    },
    "astronaut": {
        "radiation":       "crew radiation exposure",
        "communications":  "crew communication quality",
    },
    "aviation": {
        "communications":  "HF radio communications",
        "navigation":      "GPS / GNSS navigation",
    },
    "power_grid": {
        "power":           "power grid / transformer load",
        "communications":  "geomagnetic disturbance proxy (ionospheric activity)",
    },
}

# ---------------------------------------------------------------------------
# Parameter name → plain English translation
# ---------------------------------------------------------------------------

_PARAM_PLAIN_NAMES: dict[str, str] = {
    "solar_wind_speed":   "solar wind speed",
    "solar_wind_density": "solar wind density",
    "magnetic_field":     "interplanetary magnetic field (IMF)",
    "xray_flux":          "X-ray flux (solar flare activity)",
    "proton_flux":        "energetic proton flux",
    "geomagnetic_index":  "geomagnetic activity index (Kp)",
}

# ---------------------------------------------------------------------------
# Keywords that indicate a recommendation is audience-relevant
# ---------------------------------------------------------------------------

_AUDIENCE_REC_KEYWORDS: dict[str, list[str]] = {
    "satellite_operator": ["communication", "navigation", "attitude", "power", "link", "uplink",
                           "downlink", "spacecraft", "satellite", "manoeuvre", "orbit"],
    "astronaut":          ["radiation", "dose", "eva", "crew", "communication"],
    "aviation":           ["communication", "navigation", "hf", "gps", "ionospheric", "rf"],
    "power_grid":         ["power", "geomagnetic", "induced", "transformer", "grid", "communication"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_stakeholder_briefing(
    risk_assessment: dict,
    audience: str,
) -> dict[str, Any]:
    """
    Translate the output of assess_mission_risk() into audience-specific language.

    Parameters
    ----------
    risk_assessment:
        The complete dict returned by ``assess_mission_risk()``.
        Must contain at minimum the keys: "overall_risk", "domains".
    audience:
        Target audience.  Must be one of:
          "satellite_operator" | "astronaut" | "aviation" | "power_grid"

    Returns
    -------
    dict with keys:
        status           – "ok" | "error"
        audience         – the normalised audience string
        risk_summary     – 1-2 sentence plain-language summary
        relevant_domains – list of dicts with domain-specific translated risk info
        action_items     – filtered/rewritten recommendation list
        evidence_note    – one-line provenance statement
        message          – human-readable status line
    """
    # --- validate audience ---------------------------------------------------
    if not isinstance(audience, str):
        return _error(f"audience must be a string; got {type(audience).__name__}.")
    audience_norm = audience.strip().lower()
    if audience_norm not in _SUPPORTED_AUDIENCES:
        return _error(
            f"Unsupported audience '{audience}'. "
            f"Supported values: {sorted(_SUPPORTED_AUDIENCES)}."
        )

    # --- validate risk_assessment structure ----------------------------------
    if not isinstance(risk_assessment, dict):
        return _error("risk_assessment must be a dict (output of assess_mission_risk).")

    if "overall_risk" not in risk_assessment:
        return _error("risk_assessment is missing required key 'overall_risk'.")

    if "domains" not in risk_assessment:
        return _error("risk_assessment is missing required key 'domains'.")

    overall = risk_assessment.get("overall_risk", {})
    if not isinstance(overall, dict) or "level" not in overall:
        return _error("risk_assessment['overall_risk'] must contain key 'level'.")

    domains_list: list[dict] = risk_assessment.get("domains", [])
    if not isinstance(domains_list, list):
        return _error("risk_assessment['domains'] must be a list.")

    # --- extract values ------------------------------------------------------
    overall_level: str = str(overall.get("level", "LOW")).upper()
    overall_score: float | None = overall.get("score")

    # --- risk summary --------------------------------------------------------
    audience_ctx = _AUDIENCE_CONTEXT.get(audience_norm, {})
    risk_summary = audience_ctx.get(
        overall_level,
        f"Current space-weather conditions are at {overall_level} risk level.",
    )

    # --- relevant domains ----------------------------------------------------
    audience_domain_list = _AUDIENCE_DOMAINS[audience_norm]
    plain_names = _DOMAIN_PLAIN_NAMES.get(audience_norm, {})

    # Build a fast lookup from domain name → domain entry
    domain_lookup: dict[str, dict] = {
        d.get("domain", ""): d for d in domains_list if isinstance(d, dict)
    }

    relevant_domains_out: list[dict[str, Any]] = []
    for domain in audience_domain_list:
        entry = domain_lookup.get(domain)
        if entry is None:
            continue

        plain_domain_name = plain_names.get(domain, domain.replace("_", " "))
        risk_level = entry.get("risk", "LOW")
        score = entry.get("score")

        # Translate top drivers (up to 3)
        raw_drivers: list[str] = entry.get("drivers", [])
        translated_drivers = [_translate_driver(d) for d in raw_drivers[:3]]

        relevant_domains_out.append({
            "domain":       domain,
            "label":        plain_domain_name,
            "risk_level":   risk_level,
            "score":        score,
            "key_drivers":  translated_drivers,
        })

    # --- action items --------------------------------------------------------
    raw_recs: list[str] = risk_assessment.get("recommendations", [])
    action_items = _filter_recommendations(raw_recs, audience_norm)

    # --- evidence note -------------------------------------------------------
    evidence_note = _build_evidence_note(risk_assessment)

    # --- message -------------------------------------------------------------
    n_elevated = sum(
        1 for d in relevant_domains_out
        if d["risk_level"] in ("MODERATE", "HIGH", "CRITICAL")
    )
    if n_elevated == 0:
        msg = (
            f"Briefing generated for '{audience_norm}'. "
            f"All relevant domains are at LOW risk."
        )
    else:
        msg = (
            f"Briefing generated for '{audience_norm}'. "
            f"{n_elevated} relevant domain(s) at MODERATE or above."
        )

    return {
        "status":           "ok",
        "audience":         audience_norm,
        "risk_summary":     risk_summary,
        "relevant_domains": relevant_domains_out,
        "action_items":     action_items,
        "evidence_note":    evidence_note,
        "message":          msg,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _translate_driver(driver: str) -> str:
    """
    Replace technical parameter names with plain-English equivalents while
    preserving the epistemic prefix (OBSERVED / ANALYZED / PREDICTED /
    CORRELATED) and all other text.
    """
    result = driver
    for technical, plain in _PARAM_PLAIN_NAMES.items():
        # Replace underscored forms (as they appear in driver strings)
        result = result.replace(technical.replace("_", " ").title(), plain.title())
        result = result.replace(technical, plain)
    return result


def _filter_recommendations(
    recommendations: list[str],
    audience: str,
) -> list[str]:
    """
    Return recommendations relevant to the audience by keyword matching.
    If none match, return the full list (general recommendations always apply).
    """
    keywords = _AUDIENCE_REC_KEYWORDS.get(audience, [])
    if not keywords:
        return list(recommendations)

    matched = [
        r for r in recommendations
        if any(kw in r.lower() for kw in keywords)
    ]
    # Always include general (non-domain-specific) recs — identified by
    # the absence of any domain keyword from ALL audiences
    all_domain_keywords = {
        kw for kws in _AUDIENCE_REC_KEYWORDS.values() for kw in kws
    }
    general = [
        r for r in recommendations
        if not any(kw in r.lower() for kw in all_domain_keywords)
    ]

    combined = general + [r for r in matched if r not in general]
    return combined if combined else list(recommendations)


def _build_evidence_note(risk_assessment: dict) -> str:
    """
    Build a one-line provenance statement from the evidence dict in the
    risk_assessment, naming only the layers that contributed data.
    """
    evidence = risk_assessment.get("evidence", {})
    if not isinstance(evidence, dict):
        return "Evidence provenance unavailable."

    layer_names = {
        "observed":   "current conditions",
        "analyzed":   "trend and anomaly analysis",
        "predicted":  "forecast data",
        "correlated": "event correlation",
    }
    present = [
        name for key, name in layer_names.items()
        if evidence.get(key)  # non-empty list
    ]

    if not present:
        return "Assessment derived from upstream risk assessment; no detailed evidence layers were present."
    if len(present) == 1:
        return f"Assessment derived from: {present[0]}."
    return "Assessment derived from: " + ", ".join(present[:-1]) + f", and {present[-1]}."


def _error(message: str) -> dict[str, Any]:
    return {
        "status":           "error",
        "audience":         None,
        "risk_summary":     None,
        "relevant_domains": [],
        "action_items":     [],
        "evidence_note":    None,
        "message":          message,
    }
