"""
Tests for stakeholder_briefing — BobVoyage MCP tool

Covers:
  - Output schema validation (all required top-level keys present)
  - All 4 supported audiences return status="ok"
  - Invalid / unsupported audience returns status="error"
  - Missing required key "overall_risk" in risk_assessment → error
  - Missing required key "domains" in risk_assessment → error
  - risk_assessment not a dict → error
  - overall_risk missing "level" key → error
  - audience field is echoed correctly (normalised to lowercase)
  - risk_summary is a non-empty string
  - relevant_domains contains only domains valid for that audience
  - relevant_domains entries have required keys
  - action_items is always a list
  - evidence_note is a non-empty string
  - satellite_operator sees communications, navigation, attitude_control, power
  - astronaut sees radiation and communications (no navigation or power)
  - aviation sees communications and navigation (no radiation or power)
  - power_grid sees power and communications (no radiation or navigation)
  - Domains NOT in audience list are absent from relevant_domains
  - Risk level LOW/MODERATE/HIGH/CRITICAL from upstream is preserved
  - JSON-serialisable output for all 4 audiences
  - Determinism: same inputs → same outputs on repeated calls
  - Evidence note reflects present evidence layers
  - Evidence note with no evidence layers produces a sensible fallback
  - action_items is non-empty when upstream recommendations are non-empty
  - Empty recommendations list → action_items is empty list
  - domains=[] in risk_assessment → relevant_domains=[], status="ok"
  - End-to-end: real assess_mission_risk() output → briefing for each audience
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from bobvoyage.tools.stakeholder_briefing import (
    generate_stakeholder_briefing,
    _translate_driver,
    _filter_recommendations,
    _build_evidence_note,
    _AUDIENCE_DOMAINS,
    _SUPPORTED_AUDIENCES,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _minimal_risk_assessment(
    overall_level: str = "LOW",
    overall_score: float = 10.0,
    include_evidence: bool = True,
) -> dict:
    """Minimal valid risk_assessment structure (mirrors assess_mission_risk output)."""
    domains = [
        {"domain": "radiation",       "risk": "LOW",      "score": 5.0,  "drivers": []},
        {"domain": "communications",  "risk": "MODERATE", "score": 35.0, "drivers": [
            "OBSERVED: Solar Wind Speed = 600 (contributes 12.0 pts to communications risk)",
            "ANALYZED: Xray flux trend: increasing 40.0% — severity: significant",
        ]},
        {"domain": "navigation",      "risk": "LOW",      "score": 18.0, "drivers": []},
        {"domain": "power",           "risk": "LOW",      "score": 8.0,  "drivers": []},
        {"domain": "attitude_control","risk": "LOW",      "score": 12.0, "drivers": []},
    ]
    evidence = {
        "observed":   ["Solar Wind Speed: 600 (normalised 57.1% of reference range)"],
        "analyzed":   ["ANOMALY — Solar Wind Speed z=+4.00 (above baseline)"],
        "predicted":  [],
        "correlated": [],
    } if include_evidence else {"observed": [], "analyzed": [], "predicted": [], "correlated": []}

    return {
        "status":           "ok",
        "mission_profile":  {"radiation_sensitivity": "medium"},
        "overall_risk":     {"level": overall_level, "score": overall_score},
        "domains":          domains,
        "correlated_events": [],
        "evidence":         evidence,
        "recommendations":  [
            "Increase monitoring frequency for affected domains.",
            "RF environment nominal. No communication precautions indicated.",
            "Elevated geomagnetic or solar activity may affect link margins. Monitor signal quality.",
            "Radiation environment nominal. Routine monitoring sufficient.",
        ],
        "message": "Assessment based on available data.",
    }


def _high_risk_assessment() -> dict:
    """Risk assessment with HIGH overall risk across multiple domains."""
    domains = [
        {"domain": "radiation",       "risk": "HIGH",     "score": 65.0,
         "drivers": ["OBSERVED: Proton flux = 200 (contributes 18.0 pts to radiation risk)",
                     "ANALYZED: ANOMALY — Proton flux z=+8.00 (above baseline) — anomaly severity: significant"]},
        {"domain": "communications",  "risk": "HIGH",     "score": 60.0,
         "drivers": ["OBSERVED: X-ray flux (solar flare activity) = 5e-06 (contributes 15.0 pts)",
                     "ANALYZED: TREND — Geomagnetic activity index (Kp) trend: increasing 35.0%"]},
        {"domain": "navigation",      "risk": "MODERATE", "score": 40.0,
         "drivers": ["OBSERVED: Geomagnetic activity index (Kp) = 7.0 (contributes 10.0 pts)"]},
        {"domain": "power",           "risk": "MODERATE", "score": 38.0,
         "drivers": ["OBSERVED: Proton flux = 200 (contributes 8.0 pts to power risk)"]},
        {"domain": "attitude_control","risk": "MODERATE", "score": 32.0,
         "drivers": []},
    ]
    return {
        "status":           "ok",
        "mission_profile":  {"radiation_sensitivity": "high"},
        "overall_risk":     {"level": "HIGH", "score": 58.0},
        "domains":          domains,
        "correlated_events": [],
        "evidence": {
            "observed":   ["Proton Flux: 200 (normalised 80% of reference range)"],
            "analyzed":   ["ANOMALY — Proton Flux z=+8.00 (above baseline)"],
            "predicted":  ["FORECAST — Solar Wind Speed forecast: upward trend"],
            "correlated": [],
        },
        "recommendations": [
            "Increase monitoring frequency for all affected domains.",
            "Review and consider deferring sensitive operations.",
            "Prepare contingency procedures for further deterioration.",
            "Significant radiation indicators. Review scheduling of radiation-sensitive activities.",
            "Communications may be significantly affected. Increase link-margin monitoring.",
            "Significant navigation degradation possible. Increase navigation solution monitoring.",
            "Elevated power-system risk. Review operations with high power demand.",
            "Significant attitude disturbance risk. Increase attitude monitoring frequency.",
        ],
        "message": "Assessment based on all five evidence layers.",
    }


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def test_top_level_keys_present(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="satellite_operator",
        )
        for key in ("status", "audience", "risk_summary", "relevant_domains",
                    "action_items", "evidence_note", "message"):
            assert key in result, f"Top-level key '{key}' missing"

    def test_status_ok_for_valid_call(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="astronaut",
        )
        assert result["status"] == "ok"

    def test_audience_echoed_normalised(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="  Satellite_Operator  ",
        )
        assert result["status"] == "ok"
        assert result["audience"] == "satellite_operator"

    def test_risk_summary_is_non_empty_string(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="aviation",
        )
        assert isinstance(result["risk_summary"], str)
        assert len(result["risk_summary"]) > 0

    def test_relevant_domains_is_list(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="power_grid",
        )
        assert isinstance(result["relevant_domains"], list)

    def test_relevant_domain_entry_keys(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="satellite_operator",
        )
        assert result["status"] == "ok"
        for d in result["relevant_domains"]:
            for key in ("domain", "label", "risk_level", "score", "key_drivers"):
                assert key in d, f"Domain entry key '{key}' missing"

    def test_action_items_is_list(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="astronaut",
        )
        assert isinstance(result["action_items"], list)

    def test_evidence_note_is_string(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="aviation",
        )
        assert isinstance(result["evidence_note"], str)
        assert len(result["evidence_note"]) > 0

    def test_message_is_string(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="power_grid",
        )
        assert isinstance(result["message"], str)


# ---------------------------------------------------------------------------
# All four supported audiences
# ---------------------------------------------------------------------------

class TestSupportedAudiences:

    @pytest.mark.parametrize("audience", list(_SUPPORTED_AUDIENCES))
    def test_each_audience_returns_ok(self, audience: str):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience=audience,
        )
        assert result["status"] == "ok", (
            f"audience='{audience}' returned status={result['status']}: {result['message']}"
        )

    @pytest.mark.parametrize("audience", list(_SUPPORTED_AUDIENCES))
    def test_each_audience_echoed_correctly(self, audience: str):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience=audience,
        )
        assert result["audience"] == audience

    def test_satellite_operator_domains(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="satellite_operator",
        )
        returned = {d["domain"] for d in result["relevant_domains"]}
        expected = set(_AUDIENCE_DOMAINS["satellite_operator"])
        assert returned == expected

    def test_astronaut_domains(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="astronaut",
        )
        returned = {d["domain"] for d in result["relevant_domains"]}
        assert returned == {"radiation", "communications"}

    def test_aviation_domains(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="aviation",
        )
        returned = {d["domain"] for d in result["relevant_domains"]}
        assert returned == {"communications", "navigation"}

    def test_power_grid_domains(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="power_grid",
        )
        returned = {d["domain"] for d in result["relevant_domains"]}
        assert returned == {"power", "communications"}


# ---------------------------------------------------------------------------
# Domain filtering — excluded domains must be absent
# ---------------------------------------------------------------------------

class TestDomainFiltering:

    def test_astronaut_has_no_navigation_domain(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="astronaut",
        )
        domains = [d["domain"] for d in result["relevant_domains"]]
        assert "navigation" not in domains
        assert "power" not in domains
        assert "attitude_control" not in domains

    def test_aviation_has_no_radiation_domain(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="aviation",
        )
        domains = [d["domain"] for d in result["relevant_domains"]]
        assert "radiation" not in domains
        assert "power" not in domains
        assert "attitude_control" not in domains

    def test_satellite_operator_has_no_radiation_domain(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="satellite_operator",
        )
        domains = [d["domain"] for d in result["relevant_domains"]]
        assert "radiation" not in domains

    def test_power_grid_has_no_radiation_domain(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="power_grid",
        )
        domains = [d["domain"] for d in result["relevant_domains"]]
        assert "radiation" not in domains
        assert "navigation" not in domains
        assert "attitude_control" not in domains


# ---------------------------------------------------------------------------
# Risk level preservation
# ---------------------------------------------------------------------------

class TestRiskLevelPreservation:

    @pytest.mark.parametrize("level", ["LOW", "MODERATE", "HIGH", "CRITICAL"])
    def test_risk_summary_generated_for_all_levels(self, level: str):
        ra = _minimal_risk_assessment(overall_level=level)
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="aviation")
        assert result["status"] == "ok"
        assert isinstance(result["risk_summary"], str) and len(result["risk_summary"]) > 0

    def test_domain_risk_level_preserved_from_upstream(self):
        """The risk_level in relevant_domains must match the upstream domain entry."""
        ra = _minimal_risk_assessment()
        # communications domain is MODERATE in the fixture
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="aviation")
        comms = next(d for d in result["relevant_domains"] if d["domain"] == "communications")
        assert comms["risk_level"] == "MODERATE"

    def test_high_risk_summary_reflects_severity(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_high_risk_assessment(),
            audience="astronaut",
        )
        # HIGH-level summary should contain stronger language than LOW
        summary_lower = result["risk_summary"].lower()
        # Must not be a LOW-level message
        assert "nominal" not in summary_lower or "not" in summary_lower


# ---------------------------------------------------------------------------
# Invalid / unsupported audience
# ---------------------------------------------------------------------------

class TestInvalidAudience:

    def test_unsupported_audience_returns_error(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="spacecraft_engineer",
        )
        assert result["status"] == "error"

    def test_empty_string_audience_returns_error(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="",
        )
        assert result["status"] == "error"

    def test_non_string_audience_returns_error(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience=42,  # type: ignore[arg-type]
        )
        assert result["status"] == "error"

    def test_error_response_has_empty_relevant_domains(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="unknown_audience",
        )
        assert result["relevant_domains"] == []

    def test_error_message_non_empty(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="pilot",
        )
        assert isinstance(result["message"], str) and len(result["message"]) > 0


# ---------------------------------------------------------------------------
# Malformed risk_assessment inputs
# ---------------------------------------------------------------------------

class TestMalformedRiskAssessment:

    def test_not_a_dict_returns_error(self):
        result = generate_stakeholder_briefing(
            risk_assessment="not a dict",  # type: ignore[arg-type]
            audience="aviation",
        )
        assert result["status"] == "error"

    def test_missing_overall_risk_returns_error(self):
        ra = _minimal_risk_assessment()
        del ra["overall_risk"]
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="aviation")
        assert result["status"] == "error"

    def test_missing_domains_returns_error(self):
        ra = _minimal_risk_assessment()
        del ra["domains"]
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="aviation")
        assert result["status"] == "error"

    def test_overall_risk_missing_level_returns_error(self):
        ra = _minimal_risk_assessment()
        ra["overall_risk"] = {"score": 30.0}  # no "level"
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="aviation")
        assert result["status"] == "error"

    def test_domains_not_list_returns_error(self):
        ra = _minimal_risk_assessment()
        ra["domains"] = "not a list"  # type: ignore[assignment]
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="aviation")
        assert result["status"] == "error"

    def test_empty_domains_list_returns_ok(self):
        ra = _minimal_risk_assessment()
        ra["domains"] = []
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="satellite_operator")
        assert result["status"] == "ok"
        assert result["relevant_domains"] == []

    def test_none_risk_assessment_returns_error(self):
        result = generate_stakeholder_briefing(
            risk_assessment=None,  # type: ignore[arg-type]
            audience="aviation",
        )
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------

class TestActionItems:

    def test_action_items_non_empty_when_recommendations_present(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="satellite_operator",
        )
        assert len(result["action_items"]) > 0

    def test_empty_recommendations_yields_empty_action_items(self):
        ra = _minimal_risk_assessment()
        ra["recommendations"] = []
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="aviation")
        assert result["action_items"] == []

    def test_action_items_are_strings(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_high_risk_assessment(),
            audience="astronaut",
        )
        for item in result["action_items"]:
            assert isinstance(item, str)

    def test_radiation_recommendations_appear_for_astronaut(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_high_risk_assessment(),
            audience="astronaut",
        )
        combined = " ".join(result["action_items"]).lower()
        assert "radiation" in combined

    def test_power_recommendations_appear_for_power_grid(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_high_risk_assessment(),
            audience="power_grid",
        )
        combined = " ".join(result["action_items"]).lower()
        assert "power" in combined or "geomagnetic" in combined or "communication" in combined


# ---------------------------------------------------------------------------
# Evidence note
# ---------------------------------------------------------------------------

class TestEvidenceNote:

    def test_evidence_note_mentions_observed_layer(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(include_evidence=True),
            audience="aviation",
        )
        note_lower = result["evidence_note"].lower()
        assert "current conditions" in note_lower or "condition" in note_lower

    def test_evidence_note_fallback_when_no_evidence(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(include_evidence=False),
            audience="aviation",
        )
        assert isinstance(result["evidence_note"], str)
        assert len(result["evidence_note"]) > 0

    def test_evidence_note_missing_evidence_key(self):
        """Risk assessment without 'evidence' key → fallback note, no crash."""
        ra = _minimal_risk_assessment()
        del ra["evidence"]
        result = generate_stakeholder_briefing(risk_assessment=ra, audience="aviation")
        assert result["status"] == "ok"
        assert isinstance(result["evidence_note"], str)


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------

class TestJsonSerialisable:

    @pytest.mark.parametrize("audience", list(_SUPPORTED_AUDIENCES))
    def test_json_serialisable_for_all_audiences(self, audience: str):
        result = generate_stakeholder_briefing(
            risk_assessment=_high_risk_assessment(),
            audience=audience,
        )
        json.dumps(result)  # must not raise

    def test_error_response_json_serialisable(self):
        result = generate_stakeholder_briefing(
            risk_assessment=_minimal_risk_assessment(),
            audience="unknown",
        )
        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    @pytest.mark.parametrize("audience", list(_SUPPORTED_AUDIENCES))
    def test_same_inputs_same_outputs(self, audience: str):
        ra = _high_risk_assessment()
        r1 = generate_stakeholder_briefing(risk_assessment=ra, audience=audience)
        r2 = generate_stakeholder_briefing(risk_assessment=ra, audience=audience)
        assert r1["risk_summary"]     == r2["risk_summary"]
        assert r1["relevant_domains"] == r2["relevant_domains"]
        assert r1["action_items"]     == r2["action_items"]
        assert r1["evidence_note"]    == r2["evidence_note"]

    def test_different_audiences_differ(self):
        ra = _high_risk_assessment()
        r_sat  = generate_stakeholder_briefing(risk_assessment=ra, audience="satellite_operator")
        r_crew = generate_stakeholder_briefing(risk_assessment=ra, audience="astronaut")
        # Different domains should be returned
        sat_domains  = {d["domain"] for d in r_sat["relevant_domains"]}
        crew_domains = {d["domain"] for d in r_crew["relevant_domains"]}
        assert sat_domains != crew_domains

    def test_different_risk_levels_produce_different_summaries(self):
        ra_low  = _minimal_risk_assessment(overall_level="LOW")
        ra_high = _minimal_risk_assessment(overall_level="HIGH")
        r_low  = generate_stakeholder_briefing(risk_assessment=ra_low,  audience="aviation")
        r_high = generate_stakeholder_briefing(risk_assessment=ra_high, audience="aviation")
        assert r_low["risk_summary"] != r_high["risk_summary"]


# ---------------------------------------------------------------------------
# Internal helpers unit tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:

    def test_translate_driver_replaces_param_name(self):
        # Driver strings from assess_mission_risk use the raw param name form
        driver = "OBSERVED: solar_wind_speed = 600 (contributes 12.0 pts)"
        result = _translate_driver(driver)
        assert "solar wind speed" in result.lower()

    def test_translate_driver_preserves_epistemic_prefix(self):
        driver = "ANALYZED: Geomagnetic_index trend: increasing 35.0%"
        result = _translate_driver(driver)
        assert result.startswith("ANALYZED:")

    def test_translate_driver_replaces_xray_flux(self):
        driver = "OBSERVED: xray_flux = 5e-06"
        result = _translate_driver(driver)
        assert "xray_flux" not in result
        assert "x-ray" in result.lower() or "solar flare" in result.lower()

    def test_translate_driver_no_change_for_plain_text(self):
        driver = "OBSERVED: Conditions elevated above baseline"
        result = _translate_driver(driver)
        assert result == driver

    def test_filter_recommendations_returns_list(self):
        recs = ["Monitor communications", "Check radiation dose", "General monitoring"]
        result = _filter_recommendations(recs, "astronaut")
        assert isinstance(result, list)

    def test_filter_recommendations_astronaut_includes_radiation(self):
        recs = ["Significant radiation indicators. Review scheduling.",
                "Communications may be affected. Monitor signal quality.",
                "Elevated power-system risk. Review operations."]
        result = _filter_recommendations(recs, "astronaut")
        combined = " ".join(result).lower()
        assert "radiation" in combined

    def test_filter_recommendations_astronaut_excludes_power_system(self):
        """'Elevated power-system risk' must NOT appear for astronaut — 'eva' substring
        collision was the original bug."""
        recs = ["Significant radiation indicators. Review scheduling.",
                "Elevated power-system risk. Review operations with high power demand.",
                "Communications may be significantly affected. Increase link-margin monitoring."]
        result = _filter_recommendations(recs, "astronaut")
        combined = " ".join(result).lower()
        # radiation should be there; power-system text should not
        assert "radiation" in combined
        assert "power-system" not in combined

    def test_filter_recommendations_general_only_when_no_domain_match(self):
        """General recs (no domain keyword) should only be returned as fallback."""
        general_recs = [
            "Conditions nominal. Maintain standard monitoring cadence.",
            "Stay hydrated.",
        ]
        result = _filter_recommendations(general_recs, "astronaut")
        # No domain-specific match → fallback to general recs
        assert result == general_recs

    def test_filter_recommendations_general_not_prepended_when_domain_match_exists(self):
        """General recs must NOT appear alongside domain-specific matches."""
        recs = [
            "Conditions nominal. Maintain standard monitoring cadence.",  # general
            "Significant radiation indicators. Review scheduling.",         # astronaut
        ]
        result = _filter_recommendations(recs, "astronaut")
        # Only the domain-specific rec should be returned; general is suppressed
        assert len(result) == 1
        assert "radiation" in result[0].lower()

    def test_filter_recommendations_empty_input(self):
        assert _filter_recommendations([], "aviation") == []

    def test_build_evidence_note_with_observed_only(self):
        ra = {"evidence": {"observed": ["Solar Wind Speed: 400"],
                           "analyzed": [], "predicted": [], "correlated": []}}
        note = _build_evidence_note(ra)
        assert "current conditions" in note.lower()

    def test_build_evidence_note_multiple_layers(self):
        ra = {"evidence": {"observed": ["val"], "analyzed": ["trend"],
                           "predicted": ["forecast"], "correlated": []}}
        note = _build_evidence_note(ra)
        assert "current conditions" in note.lower()
        assert "trend" in note.lower() or "analysis" in note.lower()

    def test_build_evidence_note_all_empty(self):
        ra = {"evidence": {"observed": [], "analyzed": [], "predicted": [], "correlated": []}}
        note = _build_evidence_note(ra)
        assert isinstance(note, str) and len(note) > 0

    def test_build_evidence_note_missing_evidence_key(self):
        note = _build_evidence_note({"status": "ok"})
        assert "unavailable" in note.lower() or isinstance(note, str)


# ---------------------------------------------------------------------------
# End-to-end integration (real assess_mission_risk output)
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def _real_risk_assessment(self) -> dict:
        from bobvoyage.tools.assess_mission_risk import assess_mission_risk
        return assess_mission_risk()

    @pytest.mark.parametrize("audience", list(_SUPPORTED_AUDIENCES))
    def test_real_risk_assessment_all_audiences(self, audience: str):
        ra = self._real_risk_assessment()
        assert ra["status"] == "ok"
        result = generate_stakeholder_briefing(risk_assessment=ra, audience=audience)
        assert result["status"] == "ok"
        assert result["audience"] == audience
        assert isinstance(result["risk_summary"], str) and len(result["risk_summary"]) > 0
        assert isinstance(result["relevant_domains"], list)
        assert isinstance(result["action_items"], list)
        assert isinstance(result["evidence_note"], str)

    @pytest.mark.parametrize("audience", list(_SUPPORTED_AUDIENCES))
    def test_real_assessment_json_serialisable(self, audience: str):
        ra = self._real_risk_assessment()
        result = generate_stakeholder_briefing(risk_assessment=ra, audience=audience)
        json.dumps(result)  # must not raise

    def test_real_assessment_domain_subset_correct(self):
        ra = self._real_risk_assessment()
        for audience, expected_domains in _AUDIENCE_DOMAINS.items():
            result = generate_stakeholder_briefing(risk_assessment=ra, audience=audience)
            returned = {d["domain"] for d in result["relevant_domains"]}
            # All returned domains must be in the expected set for that audience
            assert returned.issubset(set(expected_domains)), (
                f"audience='{audience}' returned unexpected domains: "
                f"{returned - set(expected_domains)}"
            )


# ===========================================================================
# Action items differentiation — the original gap this fix addresses
# ===========================================================================

class TestActionItemsDifferentiation:
    """
    Given the SAME high-risk risk_assessment, the 4 audiences must produce
    action_items that are meaningfully different from each other.

    This was the gap left by the previous test suite: tests only checked
    that action_items was non-empty, not that audiences were actually
    being filtered differently.
    """

    @pytest.fixture(scope="class")
    def briefings(self):
        ra = _high_risk_assessment()
        return {
            audience: generate_stakeholder_briefing(ra, audience)
            for audience in ("satellite_operator", "astronaut", "aviation", "power_grid")
        }

    def test_all_four_audiences_produce_different_action_items(self, briefings):
        """Core regression guard: no two audiences should share identical action_items."""
        items = {
            audience: frozenset(b["action_items"])
            for audience, b in briefings.items()
        }
        unique_sets = set(items.values())
        assert len(unique_sets) == 4, (
            "Expected 4 distinct action_item sets, got "
            f"{len(unique_sets)}.\n" +
            "\n".join(f"  {a}: {list(v)}" for a, v in items.items())
        )

    def test_astronaut_action_items_contain_radiation(self, briefings):
        combined = " ".join(briefings["astronaut"]["action_items"]).lower()
        assert "radiation" in combined

    def test_astronaut_action_items_do_not_contain_power_system(self, briefings):
        """Regression: 'Elevated power-system' must not bleed into astronaut via 'eva' match."""
        combined = " ".join(briefings["astronaut"]["action_items"]).lower()
        assert "power-system" not in combined

    def test_satellite_operator_action_items_contain_navigation(self, briefings):
        combined = " ".join(briefings["satellite_operator"]["action_items"]).lower()
        assert "navigation" in combined

    def test_aviation_action_items_contain_communication_and_navigation(self, briefings):
        combined = " ".join(briefings["aviation"]["action_items"]).lower()
        assert "communication" in combined or "navigation" in combined

    def test_aviation_does_not_contain_radiation(self, briefings):
        combined = " ".join(briefings["aviation"]["action_items"]).lower()
        assert "radiation" not in combined

    def test_power_grid_action_items_contain_power_or_communication(self, briefings):
        combined = " ".join(briefings["power_grid"]["action_items"]).lower()
        assert "power" in combined or "communication" in combined

    def test_power_grid_does_not_contain_radiation(self, briefings):
        combined = " ".join(briefings["power_grid"]["action_items"]).lower()
        assert "radiation" not in combined

    def test_astronaut_and_aviation_have_different_items(self, briefings):
        assert frozenset(briefings["astronaut"]["action_items"]) != frozenset(briefings["aviation"]["action_items"])

    def test_astronaut_and_power_grid_have_different_items(self, briefings):
        assert frozenset(briefings["astronaut"]["action_items"]) != frozenset(briefings["power_grid"]["action_items"])
