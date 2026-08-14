"""
test_m8_correlation_risk.py

Tests for M8: integration of correlate_space_events output into assess_mission_risk.

Coverage:
  1.  No correlated events                           → existing behaviour unchanged
  2.  Weak correlation                               → small addend, no level change
  3.  Strong CME correlation                         → comms/navigation elevated
  4.  Strong SEP correlation                         → radiation elevated
  5.  FLR correlation                                → comms elevated
  6.  GST correlation                                → comms + navigation elevated
  7.  Multiple simultaneous events                   → addends accumulate
  8.  Event affecting multiple domains               → all affected domains reported
  9.  High mission sensitivity                       → amplifies correlation addend
  10. Low mission sensitivity                        → attenuates correlation addend
  11. Correlation contribution capped at _CORR_CAP   → hard cap enforced
  12. Double-counting prevention                     → env-saturated domain discounted
  13. Missing/None correlated_events                 → treated as empty list
  14. Existing risk behaviour unchanged when no events → scores match M5 baseline
  15. Deterministic scoring                          → repeated calls equal
  16. Evidence traceability                          → CORRELATED bucket populated
  17. Causal-language protection                     → no causal phrases in output

  SCHEMA tests:
  18. correlated_events key present in output
  19. correlated_events entry schema
  20. domain score_environmental and score_correlation keys present
  21. evidence dict has 'correlated' key

  HELPERS tests:
  22. _get_event_domain_relevance correct for known types
  23. _get_event_domain_relevance falls back to OTHER for unknown type
  24. _compute_corr_addend zero when no relevant events
  25. _compute_corr_addend applies env-saturation discount
  26. _compute_corr_addend hard cap
  27. _causal_guard_risk strips forbidden phrases
  28. _build_correlated_events_output sorted by score
"""
from __future__ import annotations

import json
import math

import pytest

from bobvoyage.tools.assess_mission_risk import (
    assess_mission_risk,
    _get_event_domain_relevance,
    _compute_corr_addend,
    _causal_guard_risk,
    _build_correlated_events_output,
    _EVENT_DOMAIN_RELEVANCE,
    _CORR_CAP,
    _CORR_SCALE,
    _OVERLAP_FACTOR,
    SENSITIVITY_MULTIPLIER,
    DEFAULT_MISSION_PROFILE,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _corr(event_type: str, score: float, event_id: str = "EV-001",
          event_time: str = "2025-07-20T08:00:00",
          observations_in_window: int = 12,
          evidence: list | None = None,
          interpretation: str = "moderate_temporal_association") -> dict:
    """Minimal correlation object that mirrors correlate_space_events output."""
    return {
        "event_type":           event_type,
        "event_id":             event_id,
        "event_time":           event_time,
        "correlation_score":    score,
        "interpretation":       interpretation,
        "observations_in_window": observations_in_window,
        "component_scores": {
            "temporal": 0.80, "anomaly": 0.70, "trend": 0.50, "event_weight": 1.00
        },
        "evidence":             evidence or [
            f"{event_type} temporally associated with elevated telemetry readings."
        ],
    }


def _nominal_conditions() -> dict:
    """Quiet sun: all parameters at or near reference minimums."""
    return {
        "solar_wind_speed":   350.0,
        "solar_wind_density": 5.0,
        "magnetic_field":     5.0,
        "xray_flux":          1e-7,
        "proton_flux":        0.1,
        "geomagnetic_index":  1.0,
    }


def _elevated_conditions() -> dict:
    """Mildly elevated: solar wind spike, elevated x-ray."""
    return {
        "solar_wind_speed":   600.0,
        "solar_wind_density": 12.0,
        "magnetic_field":     15.0,
        "xray_flux":          5e-6,
        "proton_flux":        50.0,
        "geomagnetic_index":  5.0,
    }


# ===========================================================================
# 1. No correlated events — existing behaviour unchanged
# ===========================================================================
class TestNoCorrelatedEvents:

    def test_no_events_returns_ok(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=None,
        )
        assert result["status"] == "ok"

    def test_no_events_correlated_events_key_is_empty_list(self):
        result = assess_mission_risk(conditions=_nominal_conditions())
        assert result["correlated_events"] == []

    def test_no_events_score_matches_no_correlation_baseline(self):
        base = assess_mission_risk(conditions=_nominal_conditions())
        same = assess_mission_risk(conditions=_nominal_conditions(), correlated_events=[])
        assert base["overall_risk"]["score"] == same["overall_risk"]["score"]

    def test_correlated_evidence_bucket_empty_when_no_events(self):
        result = assess_mission_risk(conditions=_nominal_conditions())
        assert result["evidence"]["correlated"] == []

    def test_domain_score_correlation_zero_when_no_events(self):
        result = assess_mission_risk(conditions=_nominal_conditions())
        for dr in result["domains"]:
            assert dr["score_correlation"] == 0.0


# ===========================================================================
# 2. Weak correlation — small addend, risk level unchanged
# ===========================================================================
class TestWeakCorrelation:

    def test_weak_cme_small_addend(self):
        corrs = [_corr("CME", 0.15, interpretation="weak_temporal_association")]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        # Addend should be positive but small
        comms = next(d for d in result["domains"] if d["domain"] == "communications")
        assert 0.0 < comms["score_correlation"] < 5.0

    def test_weak_correlation_does_not_escalate_level(self):
        base   = assess_mission_risk(conditions=_nominal_conditions())
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.1, interpretation="no_significant_correlation")],
        )
        # Overall level must not jump by more than one tier from a very weak event
        levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        base_idx   = levels.index(base["overall_risk"]["level"])
        result_idx = levels.index(result["overall_risk"]["level"])
        assert result_idx - base_idx <= 1


# ===========================================================================
# 3. Strong CME — communications and navigation elevated
# ===========================================================================
class TestStrongCMECorrelation:

    def test_cme_increases_communications_score(self):
        base   = assess_mission_risk(conditions=_nominal_conditions())
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.90, interpretation="strong_temporal_association")],
        )
        base_comms   = next(d["score"] for d in base["domains"]   if d["domain"] == "communications")
        result_comms = next(d["score"] for d in result["domains"] if d["domain"] == "communications")
        assert result_comms > base_comms

    def test_cme_increases_navigation_score(self):
        base   = assess_mission_risk(conditions=_nominal_conditions())
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.90)],
        )
        base_nav   = next(d["score"] for d in base["domains"]   if d["domain"] == "navigation")
        result_nav = next(d["score"] for d in result["domains"] if d["domain"] == "navigation")
        assert result_nav > base_nav

    def test_cme_with_nominal_env_has_comms_contribution(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 1.0)],
        )
        comms = next(d for d in result["domains"] if d["domain"] == "communications")
        assert comms["score_correlation"] > 0

    def test_strong_cme_appears_in_correlated_events_output(self):
        corrs  = [_corr("CME", 0.85)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        assert len(result["correlated_events"]) == 1
        assert result["correlated_events"][0]["event_type"] == "CME"
        assert result["correlated_events"][0]["correlation_score"] == pytest.approx(0.85)


# ===========================================================================
# 4. Strong SEP — radiation elevated
# ===========================================================================
class TestStrongSEPCorrelation:

    def test_sep_increases_radiation_score(self):
        base   = assess_mission_risk(conditions=_nominal_conditions())
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("SEP", 0.90)],
        )
        base_rad   = next(d["score"] for d in base["domains"]   if d["domain"] == "radiation")
        result_rad = next(d["score"] for d in result["domains"] if d["domain"] == "radiation")
        assert result_rad > base_rad

    def test_sep_radiation_relevance_highest(self):
        sep_rad  = _get_event_domain_relevance("SEP", "radiation")
        cme_rad  = _get_event_domain_relevance("CME", "radiation")
        flr_rad  = _get_event_domain_relevance("FLR", "radiation")
        # SEP must have the highest radiation relevance
        assert sep_rad > cme_rad
        assert sep_rad > flr_rad

    def test_sep_radiation_contribution_in_correlated_events_output(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("SEP", 0.90)],
        )
        ce = result["correlated_events"][0]
        assert "radiation" in ce["affected_domains"]


# ===========================================================================
# 5. FLR — communications elevated
# ===========================================================================
class TestFLRCorrelation:

    def test_flr_increases_communications_score(self):
        base   = assess_mission_risk(conditions=_nominal_conditions())
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("FLR", 0.80)],
        )
        base_comms   = next(d["score"] for d in base["domains"]   if d["domain"] == "communications")
        result_comms = next(d["score"] for d in result["domains"] if d["domain"] == "communications")
        assert result_comms > base_comms

    def test_flr_communications_relevance_high(self):
        assert _get_event_domain_relevance("FLR", "communications") >= 0.85

    def test_flr_attitude_relevance_low(self):
        assert _get_event_domain_relevance("FLR", "attitude_control") < 0.15


# ===========================================================================
# 6. GST — communications and navigation elevated
# ===========================================================================
class TestGSTCorrelation:

    def test_gst_increases_comms_and_navigation(self):
        base   = assess_mission_risk(conditions=_nominal_conditions())
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("GST", 0.85)],
        )
        for domain in ("communications", "navigation"):
            base_s   = next(d["score"] for d in base["domains"]   if d["domain"] == domain)
            result_s = next(d["score"] for d in result["domains"] if d["domain"] == domain)
            assert result_s > base_s, f"{domain} score should increase with GST"

    def test_gst_navigation_relevance_among_highest(self):
        gst_nav = _get_event_domain_relevance("GST", "navigation")
        assert gst_nav >= 0.80

    def test_gst_attitude_relevance_significant(self):
        assert _get_event_domain_relevance("GST", "attitude_control") >= 0.60


# ===========================================================================
# 7. Multiple simultaneous events — addends accumulate
# ===========================================================================
class TestMultipleSimultaneousEvents:

    def test_two_events_higher_than_one(self):
        single = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.70)],
        )
        double = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.70), _corr("GST", 0.70, event_id="EV-002")],
        )
        single_comms = next(d["score"] for d in single["domains"] if d["domain"] == "communications")
        double_comms = next(d["score"] for d in double["domains"] if d["domain"] == "communications")
        assert double_comms >= single_comms

    def test_multiple_events_all_appear_in_correlated_events_section(self):
        corrs = [
            _corr("CME", 0.80),
            _corr("SEP", 0.70, event_id="EV-002"),
            _corr("FLR", 0.60, event_id="EV-003"),
        ]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        assert len(result["correlated_events"]) == 3

    def test_multiple_events_sorted_by_score_descending(self):
        corrs = [
            _corr("FLR", 0.50, event_id="EV-A"),
            _corr("CME", 0.90, event_id="EV-B"),
            _corr("GST", 0.70, event_id="EV-C"),
        ]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        scores = [ce["correlation_score"] for ce in result["correlated_events"]]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 8. Event affecting multiple domains
# ===========================================================================
class TestEventMultipleDomains:

    def test_cme_affects_multiple_domains(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.90)],
        )
        ce = result["correlated_events"][0]
        # CME should affect at least 3 domains
        assert len(ce["affected_domains"]) >= 3

    def test_cme_risk_contribution_dict_non_empty(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.90)],
        )
        ce = result["correlated_events"][0]
        assert len(ce["risk_contribution"]) > 0
        for domain, val in ce["risk_contribution"].items():
            assert val >= 0.5

    def test_sep_affects_radiation_and_power(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("SEP", 0.90)],
        )
        ce = result["correlated_events"][0]
        assert "radiation" in ce["affected_domains"]
        assert "power" in ce["affected_domains"]


# ===========================================================================
# 9. High mission sensitivity amplifies correlation addend
# ===========================================================================
class TestHighSensitivityAmplification:

    def test_high_sensitivity_yields_larger_addend_than_low(self):
        high_result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("SEP", 0.80)],
            mission_profile={
                **DEFAULT_MISSION_PROFILE,
                "radiation_sensitivity": "high",
            },
        )
        low_result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("SEP", 0.80)],
            mission_profile={
                **DEFAULT_MISSION_PROFILE,
                "radiation_sensitivity": "low",
            },
        )
        high_rad = next(d["score_correlation"] for d in high_result["domains"] if d["domain"] == "radiation")
        low_rad  = next(d["score_correlation"] for d in low_result["domains"]  if d["domain"] == "radiation")
        assert high_rad > low_rad

    def test_high_sensitivity_may_escalate_risk_level(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.95)],
            mission_profile={
                **DEFAULT_MISSION_PROFILE,
                "communications_sensitivity": "high",
            },
        )
        comms = next(d for d in result["domains"] if d["domain"] == "communications")
        # With score 0.95 CME + high sensitivity, comms should reach at least MODERATE
        assert comms["risk"] in ("MODERATE", "HIGH", "CRITICAL")


# ===========================================================================
# 10. Low mission sensitivity attenuates correlation addend
# ===========================================================================
class TestLowSensitivityAttenuation:

    def test_low_sensitivity_smaller_addend(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.80)],
            mission_profile={
                **DEFAULT_MISSION_PROFILE,
                "communications_sensitivity": "low",
            },
        )
        comms = next(d for d in result["domains"] if d["domain"] == "communications")
        assert comms["score_correlation"] < _CORR_CAP

    def test_low_sensitivity_addend_less_than_high(self):
        high = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.80)],
            mission_profile={**DEFAULT_MISSION_PROFILE, "communications_sensitivity": "high"},
        )
        low = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.80)],
            mission_profile={**DEFAULT_MISSION_PROFILE, "communications_sensitivity": "low"},
        )
        high_c = next(d["score_correlation"] for d in high["domains"] if d["domain"] == "communications")
        low_c  = next(d["score_correlation"] for d in low["domains"]  if d["domain"] == "communications")
        assert high_c > low_c


# ===========================================================================
# 11. Correlation contribution capped at _CORR_CAP
# ===========================================================================
class TestCorrelationCap:

    def test_many_perfect_score_events_capped(self):
        """Ten perfect-score CMEs in the same domain must not exceed _CORR_CAP."""
        corrs = [
            _corr("CME", 1.0, event_id=f"EV-{i:03d}")
            for i in range(10)
        ]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        for dr in result["domains"]:
            assert dr["score_correlation"] <= _CORR_CAP + 1e-6

    def test_single_perfect_event_below_cap_for_domain_without_env(self):
        """
        A single perfect-score event on a domain with zero environmental signal
        should yield base * discount.  The base is:
            1.0 × relevance × sensitivity_multiplier × _CORR_SCALE
        For CME/communications with medium sensitivity:
            1.0 × 0.90 × 1.0 × 40 = 36 pts
        With env_saturation=0 (no env signal), discount=1.0 → 36 pts ≤ _CORR_CAP=25 → capped.
        """
        addend, _ = _compute_corr_addend(
            correlations=[_corr("CME", 1.0)],
            domain="communications",
            raw_domain_score=0.0,
            multiplier=1.0,
        )
        assert addend <= _CORR_CAP


# ===========================================================================
# 12. Double-counting prevention
# ===========================================================================
class TestDoubleCountingPrevention:

    def test_high_env_score_reduces_corr_addend(self):
        """When environmental evidence is strong (high raw_domain_score), the
        correlation addend should be smaller than when env evidence is absent."""
        addend_no_env, _ = _compute_corr_addend(
            correlations=[_corr("CME", 0.80)],
            domain="communications",
            raw_domain_score=0.0,
            multiplier=1.0,
        )
        addend_full_env, _ = _compute_corr_addend(
            correlations=[_corr("CME", 0.80)],
            domain="communications",
            raw_domain_score=100.0,   # fully saturated
            multiplier=1.0,
        )
        assert addend_no_env > addend_full_env

    def test_fully_saturated_domain_still_gets_nonzero_residual(self):
        """At env_saturation=1.0, the residual (1 - _OVERLAP_FACTOR) = 0.30 should
        ensure a nonzero contribution.  This provides event-type context even when
        environmental data is already rich."""
        addend, _ = _compute_corr_addend(
            correlations=[_corr("CME", 1.0)],
            domain="communications",
            raw_domain_score=100.0,
            multiplier=1.0,
        )
        # residual = 1 - 1.0 × 0.70 = 0.30
        # base = 1.0 × 0.90 × 1.0 × 40 = 36.0 → before cap
        # discounted = 36 × 0.30 = 10.8 → ≤ 25 (cap), so should be ~10.8
        assert addend > 0.0

    def test_env_saturation_formula_correct(self):
        """Manually verify the discount formula at a specific saturation level."""
        # raw_domain_score = 25 → sat = 0.5 → discount = 1 - 0.5 × 0.70 = 0.65
        score = 0.80
        relevance = _get_event_domain_relevance("CME", "communications")  # 0.90
        multiplier = 1.0
        expected_base = score * relevance * multiplier * _CORR_SCALE
        expected_discount = 1.0 - (25.0 / 50.0) * _OVERLAP_FACTOR
        expected_discounted = expected_base * expected_discount

        addend, details = _compute_corr_addend(
            correlations=[_corr("CME", score)],
            domain="communications",
            raw_domain_score=25.0,
            multiplier=multiplier,
        )
        # Result should be capped at _CORR_CAP or equal to expected_discounted
        assert addend == pytest.approx(min(expected_discounted, _CORR_CAP), abs=1e-6)


# ===========================================================================
# 13. Missing / None correlated_events handled gracefully
# ===========================================================================
class TestMissingCorrelatedEvents:

    def test_none_correlated_events_ok(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=None,
        )
        assert result["status"] == "ok"

    def test_empty_list_correlated_events_ok(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[],
        )
        assert result["status"] == "ok"
        assert result["correlated_events"] == []

    def test_none_correlated_events_no_correlated_evidence(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=None,
        )
        assert result["evidence"]["correlated"] == []


# ===========================================================================
# 14. Existing risk behaviour unchanged when no events
# ===========================================================================
class TestExistingBehaviourUnchanged:

    def test_no_events_overall_score_matches_legacy_call(self):
        """assess_mission_risk without correlated_events must behave exactly as before."""
        legacy = assess_mission_risk(
            conditions=_elevated_conditions(),
            anomalies=[
                {"parameter": "solar_wind_speed", "severity": "significant",
                 "z_score": 3.2, "direction": "above_baseline"},
            ],
        )
        new = assess_mission_risk(
            conditions=_elevated_conditions(),
            anomalies=[
                {"parameter": "solar_wind_speed", "severity": "significant",
                 "z_score": 3.2, "direction": "above_baseline"},
            ],
            correlated_events=None,
        )
        assert legacy["overall_risk"]["score"] == new["overall_risk"]["score"]

    def test_no_events_domain_scores_match_legacy(self):
        """Domain scores must be identical with and without explicit empty events."""
        legacy = assess_mission_risk(conditions=_nominal_conditions())
        new    = assess_mission_risk(conditions=_nominal_conditions(), correlated_events=[])
        for ld, nd in zip(legacy["domains"], new["domains"]):
            assert ld["score"] == nd["score"]


# ===========================================================================
# 15. Deterministic scoring
# ===========================================================================
class TestDeterminism:

    def test_same_events_same_scores(self):
        corrs = [_corr("CME", 0.75), _corr("SEP", 0.60, event_id="EV-002")]
        r1 = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        r2 = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        assert r1["overall_risk"]["score"] == r2["overall_risk"]["score"]
        for d1, d2 in zip(r1["domains"], r2["domains"]):
            assert d1["score"] == d2["score"]
            assert d1["score_correlation"] == d2["score_correlation"]

    def test_output_is_json_serialisable(self):
        corrs = [_corr("GST", 0.80)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        serialised = json.loads(json.dumps(result))
        assert serialised["status"] == "ok"


# ===========================================================================
# 16. Evidence traceability
# ===========================================================================
class TestEvidenceTraceability:

    def test_correlated_evidence_bucket_populated(self):
        corrs = [_corr("CME", 0.80)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        assert len(result["evidence"]["correlated"]) >= 1

    def test_correlated_evidence_mentions_event_type(self):
        corrs = [_corr("SEP", 0.75)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        combined = " ".join(result["evidence"]["correlated"]).upper()
        assert "SEP" in combined

    def test_correlated_domain_drivers_present_in_domain(self):
        corrs = [_corr("CME", 0.85)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        comms = next(d for d in result["domains"] if d["domain"] == "communications")
        corr_drivers = [dr for dr in comms["drivers"] if dr.startswith("CORRELATED:")]
        assert len(corr_drivers) >= 1

    def test_all_four_evidence_keys_present(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("FLR", 0.70)],
        )
        for key in ("observed", "analyzed", "predicted", "correlated"):
            assert key in result["evidence"]

    def test_domain_has_score_environmental_and_score_correlation(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.70)],
        )
        for dr in result["domains"]:
            assert "score_environmental" in dr
            assert "score_correlation" in dr
            assert dr["score"] == pytest.approx(
                dr["score_environmental"] + dr["score_correlation"], abs=0.2
            )

    def test_correlated_events_output_has_required_keys(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("CME", 0.80)],
        )
        ce = result["correlated_events"][0]
        for key in ("event_type", "event_id", "event_time", "correlation_score",
                    "interpretation", "affected_domains", "risk_contribution",
                    "observations_in_window", "evidence"):
            assert key in ce, f"Missing key: {key}"


# ===========================================================================
# 17. Causal-language protection
# ===========================================================================
class TestCausalLanguageProtection:

    def test_no_caused_in_correlated_evidence(self):
        raw_evidence = ["The CME caused the observed solar wind spike."]
        corrs = [_corr("CME", 0.80, evidence=raw_evidence)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        ce = result["correlated_events"][0]
        all_evidence = " ".join(ce["evidence"]).lower()
        assert "caused" not in all_evidence

    def test_no_due_to_in_correlated_evidence(self):
        raw_evidence = ["Anomaly due to the CME event."]
        corrs = [_corr("CME", 0.80, evidence=raw_evidence)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        ce = result["correlated_events"][0]
        all_evidence = " ".join(ce["evidence"]).lower()
        assert "due to" not in all_evidence

    def test_no_resulted_in_correlated_evidence(self):
        raw_evidence = ["The SEP event resulted in a proton flux increase."]
        corrs = [_corr("SEP", 0.80, evidence=raw_evidence)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        ce = result["correlated_events"][0]
        all_evidence = " ".join(ce["evidence"]).lower()
        assert "resulted in" not in all_evidence

    def test_no_causal_language_in_correlated_bucket(self):
        raw_evidence = [
            "This CME led to a geomagnetic storm.",
            "The storm was triggered by the solar event.",
        ]
        corrs = [_corr("CME", 0.80, evidence=raw_evidence)]
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=corrs,
        )
        all_corr = " ".join(result["evidence"]["correlated"]).lower()
        for phrase in ("caused", "due to", "resulted in", "led to", "triggered by"):
            assert phrase not in all_corr, f"Forbidden phrase found: '{phrase}'"

    def test_causal_guard_risk_strips_phrase(self):
        text = "The CME caused the anomaly."
        cleaned = _causal_guard_risk(text)
        assert "caused" not in cleaned.lower()

    def test_causal_guard_risk_passthrough_clean_text(self):
        text = "The CME is temporally associated with the observed disturbance."
        assert _causal_guard_risk(text) == text


# ===========================================================================
# 18–21. Output schema tests
# ===========================================================================
class TestOutputSchema:

    def test_correlated_events_key_present(self):
        result = assess_mission_risk()
        assert "correlated_events" in result

    def test_correlated_events_is_list(self):
        result = assess_mission_risk()
        assert isinstance(result["correlated_events"], list)

    def test_evidence_has_correlated_key(self):
        result = assess_mission_risk()
        assert "correlated" in result["evidence"]

    def test_domain_has_score_environmental(self):
        result = assess_mission_risk()
        for dr in result["domains"]:
            assert "score_environmental" in dr

    def test_domain_has_score_correlation(self):
        result = assess_mission_risk()
        for dr in result["domains"]:
            assert "score_correlation" in dr

    def test_error_response_has_correlated_events(self):
        result = assess_mission_risk(
            mission_profile={"radiation_sensitivity": "invalid_value"}
        )
        assert result["status"] == "error"
        assert "correlated_events" in result
        assert "correlated" in result["evidence"]


# ===========================================================================
# 22–28. Internal helper tests
# ===========================================================================
class TestInternalHelpers:

    def test_get_event_domain_relevance_cme_comms(self):
        assert _get_event_domain_relevance("CME", "communications") == pytest.approx(0.90)

    def test_get_event_domain_relevance_sep_radiation(self):
        assert _get_event_domain_relevance("SEP", "radiation") == pytest.approx(0.95)

    def test_get_event_domain_relevance_flr_comms(self):
        assert _get_event_domain_relevance("FLR", "communications") == pytest.approx(0.90)

    def test_get_event_domain_relevance_gst_navigation(self):
        assert _get_event_domain_relevance("GST", "navigation") == pytest.approx(0.85)

    def test_get_event_domain_relevance_unknown_type_returns_other(self):
        val = _get_event_domain_relevance("UNKNOWN_XRAY_BURST", "communications")
        other_val = _EVENT_DOMAIN_RELEVANCE["OTHER"]["communications"]
        assert val == pytest.approx(other_val)

    def test_get_event_domain_relevance_case_insensitive(self):
        assert _get_event_domain_relevance("cme", "communications") == \
               _get_event_domain_relevance("CME", "communications")

    def test_compute_corr_addend_zero_when_no_events(self):
        addend, details = _compute_corr_addend(
            correlations=[],
            domain="radiation",
            raw_domain_score=0.0,
            multiplier=1.0,
        )
        assert addend == 0.0
        assert details == []

    def test_compute_corr_addend_nonzero_for_cme(self):
        addend, _ = _compute_corr_addend(
            correlations=[_corr("CME", 0.80)],
            domain="communications",
            raw_domain_score=0.0,
            multiplier=1.0,
        )
        assert addend > 0.0

    def test_compute_corr_addend_discount_applied(self):
        # With raw_domain_score=50 → env_saturation=1.0
        # discount = 1 - 1.0 × 0.70 = 0.30
        addend_zero_env, _ = _compute_corr_addend(
            correlations=[_corr("CME", 0.80)],
            domain="communications",
            raw_domain_score=0.0,
            multiplier=1.0,
        )
        addend_full_env, _ = _compute_corr_addend(
            correlations=[_corr("CME", 0.80)],
            domain="communications",
            raw_domain_score=50.0,
            multiplier=1.0,
        )
        assert addend_full_env < addend_zero_env

    def test_compute_corr_addend_hard_cap(self):
        # 100 identical perfect-score CMEs — result must not exceed _CORR_CAP
        corrs = [_corr("CME", 1.0, event_id=f"EV-{i}") for i in range(100)]
        addend, _ = _compute_corr_addend(
            correlations=corrs,
            domain="communications",
            raw_domain_score=0.0,
            multiplier=1.0,
        )
        assert addend <= _CORR_CAP

    def test_build_correlated_events_output_sorted(self):
        corrs = [
            _corr("FLR", 0.40, event_id="EV-A"),
            _corr("CME", 0.90, event_id="EV-B"),
            _corr("GST", 0.65, event_id="EV-C"),
        ]
        output = _build_correlated_events_output(
            correlations=corrs,
            domain_results_raw={d: 0.0 for d in ("radiation", "communications",
                                                   "navigation", "power", "attitude_control")},
            profile=dict(DEFAULT_MISSION_PROFILE),
            profile_key={
                "radiation":       "radiation_sensitivity",
                "communications":  "communications_sensitivity",
                "navigation":      "navigation_sensitivity",
                "power":           "power_sensitivity",
                "attitude_control":"attitude_control_sensitivity",
            },
        )
        scores = [c["correlation_score"] for c in output]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 29. Badge event_type matches event["type"] from correlate_space_events output
#
# Regression test for the bug where _build_correlated_events_output read
# corr.get("event_type", "OTHER") instead of corr["event"]["type"], causing
# every badge to show "OTHER" while the evidence text showed the real type.
# ===========================================================================
class TestBadgeEventTypeMatchesRealEventType:
    """The event_type field surfaced in correlated_events (used by the dashboard
    badge) must equal the type nested inside the correlation's 'event' sub-dict,
    exactly as returned by correlate_space_events()."""

    def _make_corr_from_correlate_output(self, event_type: str, score: float) -> dict:
        """Construct a correlation dict in the exact shape that
        correlate_space_events._build_correlation() returns."""
        return {
            # NO top-level 'event_type' — the real field lives inside 'event'
            "event": {
                "type":        event_type,
                "event_time":  "2025-07-20T08:00:00+00:00",
                "external_id": f"EV-{event_type}-001",
                "source":      "NASA_DONKI",
                "severity":    "M2.1" if event_type == "FLR" else "moderate",
                "description": f"Test {event_type} event",
            },
            "observations_in_window": 5,
            "temporal_distance_minutes": 12.0,
            "top_deviations": [],
            "correlation_score": score,
            "interpretation":   "moderate_temporal_association",
            "component_scores": {
                "temporal": 0.70, "anomaly": 0.60, "trend": 0.30, "event_weight": 0.80,
            },
            "evidence": [
                f"OBSERVED: {event_type} event has reported severity: M2.1."
            ],
        }

    def test_cme_badge_matches_nested_event_type(self):
        corr = self._make_corr_from_correlate_output("CME", 0.80)
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[corr],
        )
        ce = result["correlated_events"][0]
        assert ce["event_type"] == "CME", (
            f"Badge shows '{ce['event_type']}' but event['type'] is 'CME'"
        )

    def test_flr_badge_matches_nested_event_type(self):
        corr = self._make_corr_from_correlate_output("FLR", 0.75)
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[corr],
        )
        ce = result["correlated_events"][0]
        assert ce["event_type"] == "FLR", (
            f"Badge shows '{ce['event_type']}' but event['type'] is 'FLR'"
        )

    def test_gst_badge_matches_nested_event_type(self):
        corr = self._make_corr_from_correlate_output("GST", 0.70)
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[corr],
        )
        ce = result["correlated_events"][0]
        assert ce["event_type"] == "GST"

    def test_sep_badge_matches_nested_event_type(self):
        corr = self._make_corr_from_correlate_output("SEP", 0.85)
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[corr],
        )
        ce = result["correlated_events"][0]
        assert ce["event_type"] == "SEP"

    def test_badge_not_other_when_real_type_is_cme(self):
        """Explicit guard against the original bug: badge must NOT be 'OTHER'
        when the nested event type is 'CME'."""
        corr = self._make_corr_from_correlate_output("CME", 0.80)
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[corr],
        )
        ce = result["correlated_events"][0]
        assert ce["event_type"] != "OTHER", (
            "Badge is 'OTHER' but real event type is 'CME' — nested-field bug not fixed"
        )

    def test_badge_not_other_when_real_type_is_flr(self):
        corr = self._make_corr_from_correlate_output("FLR", 0.75)
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[corr],
        )
        ce = result["correlated_events"][0]
        assert ce["event_type"] != "OTHER", (
            "Badge is 'OTHER' but real event type is 'FLR' — nested-field bug not fixed"
        )

    def test_event_type_consistent_with_evidence_text(self):
        """The badge event_type and the evidence text must name the same event type."""
        for etype in ("CME", "FLR", "GST", "SEP"):
            corr = self._make_corr_from_correlate_output(etype, 0.75)
            result = assess_mission_risk(
                conditions=_nominal_conditions(),
                correlated_events=[corr],
            )
            ce = result["correlated_events"][0]
            badge_type = ce["event_type"]
            # The first evidence entry injected above mentions the real event type
            evidence_text = " ".join(ce["evidence"]).upper()
            assert badge_type in evidence_text, (
                f"Badge says '{badge_type}' but '{etype}' evidence text doesn't contain it. "
                f"Evidence: {ce['evidence']}"
            )

    def test_event_id_and_event_time_from_nested_event(self):
        """event_id and event_time in the output must come from event.external_id
        and event.event_time, not from top-level keys that don't exist."""
        corr = self._make_corr_from_correlate_output("CME", 0.80)
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[corr],
        )
        ce = result["correlated_events"][0]
        assert ce["event_id"] == "EV-CME-001"
        assert ce["event_time"] == "2025-07-20T08:00:00+00:00"


# ===========================================================================
# 30. HSS — High Speed Stream has its own entry, not the OTHER fallback
# ===========================================================================
class TestHSSEventType:
    """HSS was previously unrecognised and fell through to the 'OTHER' fallback
    (radiation 0.05, communications 0.10, …).  These tests confirm it now has
    dedicated weights that correctly reflect its sustained-wind physical impact.

    Dataset context: 103 of 1743 events (~6%) are HSS — a material fraction
    that was silently underweighted before this fix.
    """

    def test_hss_domain_relevance_not_equal_to_other_fallback(self):
        """Every domain relevance for HSS must differ from the OTHER fallback."""
        other = _EVENT_DOMAIN_RELEVANCE["OTHER"]
        hss   = _EVENT_DOMAIN_RELEVANCE["HSS"]
        for domain in ("radiation", "communications", "navigation",
                       "power", "attitude_control"):
            assert hss[domain] != other[domain], (
                f"HSS[{domain}] == OTHER[{domain}] == {other[domain]}; "
                "HSS must have its own dedicated weight"
            )

    def test_hss_communications_relevance_higher_than_other(self):
        """HSS comms relevance (0.60) must be higher than OTHER (0.10)."""
        assert _get_event_domain_relevance("HSS", "communications") > \
               _get_event_domain_relevance("OTHER", "communications")

    def test_hss_attitude_control_relevance_higher_than_other(self):
        """HSS attitude_control relevance (0.65) must be higher than OTHER (0.05)."""
        assert _get_event_domain_relevance("HSS", "attitude_control") > \
               _get_event_domain_relevance("OTHER", "attitude_control")

    def test_hss_radiation_relevance_is_low(self):
        """HSS is not a particle event; radiation relevance must stay below 0.30."""
        assert _get_event_domain_relevance("HSS", "radiation") < 0.30

    def test_hss_attitude_control_relevance_equals_0_65(self):
        """Explicit value check: attitude_control must be 0.65."""
        assert _get_event_domain_relevance("HSS", "attitude_control") == pytest.approx(0.65)

    def test_hss_communications_relevance_equals_0_60(self):
        """Explicit value check: communications must be 0.60."""
        assert _get_event_domain_relevance("HSS", "communications") == pytest.approx(0.60)

    def test_hss_increases_communications_score(self):
        """A correlated HSS event must raise the communications domain score."""
        base   = assess_mission_risk(conditions=_nominal_conditions())
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("HSS", 0.80, interpretation="strong_temporal_association")],
        )
        base_comms   = next(d["score"] for d in base["domains"]   if d["domain"] == "communications")
        result_comms = next(d["score"] for d in result["domains"] if d["domain"] == "communications")
        assert result_comms > base_comms

    def test_hss_increases_attitude_control_score(self):
        """A correlated HSS must raise attitude_control (sustained wind pressure)."""
        base   = assess_mission_risk(conditions=_nominal_conditions())
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("HSS", 0.80)],
        )
        base_att   = next(d["score"] for d in base["domains"]   if d["domain"] == "attitude_control")
        result_att = next(d["score"] for d in result["domains"] if d["domain"] == "attitude_control")
        assert result_att > base_att

    def test_hss_domain_scores_higher_than_other_fallback(self):
        """With identical correlation score, HSS must yield higher domain scores
        than an unknown event type that falls back to OTHER weights."""
        hss_result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("HSS", 0.80)],
        )
        other_result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("UNKNOWN_RANDOM_TYPE", 0.80)],
        )
        for domain in ("communications", "navigation", "attitude_control"):
            hss_score   = next(d["score"] for d in hss_result["domains"]   if d["domain"] == domain)
            other_score = next(d["score"] for d in other_result["domains"] if d["domain"] == domain)
            assert hss_score > other_score, (
                f"{domain}: HSS score ({hss_score}) must exceed OTHER fallback score ({other_score})"
            )

    def test_hss_appears_in_correlated_events_output_with_correct_type(self):
        """The correlated_events output section must show event_type='HSS'."""
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            correlated_events=[_corr("HSS", 0.75)],
        )
        assert len(result["correlated_events"]) == 1
        assert result["correlated_events"][0]["event_type"] == "HSS"
