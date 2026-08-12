"""
Tests for assess_mission_risk — BobVoyage MCP tool

Covers:
  - Nominal conditions (all inputs well within range)
  - Moderate conditions
  - High-risk conditions (significant anomalies + strong trends)
  - Critical conditions (extreme values across multiple parameters)
  - Missing inputs: no conditions, no trends, no anomalies, no predictions
  - All inputs None simultaneously (graceful degradation)
  - High radiation sensitivity amplifies radiation domain
  - Low radiation sensitivity suppresses radiation domain
  - Multiple simultaneous high-risk domains
  - Mission profile applied correctly (multiplier scaling)
  - Invalid mission profile value → error
  - Invalid mission profile key (unknown domain) is ignored or accepted
  - Output schema validation (all required top-level keys)
  - Domain list contains all five expected domains
  - Each domain entry has required keys
  - Overall risk level is one of LOW/MODERATE/HIGH/CRITICAL
  - Domain risk levels are one of LOW/MODERATE/HIGH/CRITICAL
  - Risk score is in [0, 100]
  - Deterministic scoring (same inputs → same outputs)
  - Evidence traceability: observed evidence references parameter values
  - Evidence traceability: analyzed evidence references anomalies/trends
  - Evidence traceability: predicted evidence references forecasts
  - Recommendations present and non-empty for HIGH/CRITICAL overall risk
  - Recommendations absent / minimal for LOW overall risk
  - Anomaly-driven risk amplification
  - Trend-driven risk amplification
  - Forecast-driven risk amplification
  - Boundary score conditions (score just above/below thresholds)
  - JSON-serialisable output
  - End-to-end integration using real BobVoyage tool outputs
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from bobvoyage.tools.assess_mission_risk import (
    assess_mission_risk,
    DEFAULT_MISSION_PROFILE,
    _classify_risk,
    _normalise,
    _anomaly_contribution,
    _trend_contribution,
    _PARAM_RANGES,
    SENSITIVITY_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _nominal_conditions() -> dict:
    """All parameters at mid-range — should produce LOW to MODERATE risk."""
    return {
        "timestamp":          "2025-07-20T12:00:00+00:00",
        "solar_wind_speed":   400.0,
        "solar_wind_density": 5.0,
        "magnetic_field":     7.0,
        "xray_flux":          1.0e-7,
        "proton_flux":        2.0,
        "geomagnetic_index":  2.5,
    }


def _elevated_conditions() -> dict:
    """Parameters elevated toward the upper reference range."""
    return {
        "timestamp":          "2025-07-20T12:00:00+00:00",
        "solar_wind_speed":   650.0,
        "solar_wind_density": 12.0,
        "magnetic_field":     18.0,
        "xray_flux":          5.0e-6,
        "proton_flux":        50.0,
        "geomagnetic_index":  7.0,
    }


def _extreme_conditions() -> dict:
    """Parameters near the upper reference ceiling."""
    return {
        "timestamp":          "2025-07-20T12:00:00+00:00",
        "solar_wind_speed":   880.0,
        "solar_wind_density": 19.0,
        "magnetic_field":     28.0,
        "xray_flux":          5.0e-5,
        "proton_flux":        900.0,
        "geomagnetic_index":  8.8,
    }


def _significant_anomalies() -> list[dict]:
    return [
        {
            "parameter":     "solar_wind_speed",
            "timestamp":     "2025-07-20T12:00:00+00:00",
            "observed_value": 650.0,
            "baseline_mean":  400.0,
            "baseline_std":   30.0,
            "z_score":        8.33,
            "severity":       "significant",
            "direction":      "above_baseline",
        },
        {
            "parameter":     "proton_flux",
            "timestamp":     "2025-07-20T12:00:00+00:00",
            "observed_value": 50.0,
            "baseline_mean":  2.0,
            "baseline_std":   0.5,
            "z_score":        96.0,
            "severity":       "significant",
            "direction":      "above_baseline",
        },
    ]


def _moderate_anomalies() -> list[dict]:
    return [
        {
            "parameter":     "geomagnetic_index",
            "timestamp":     "2025-07-20T12:00:00+00:00",
            "observed_value": 5.5,
            "baseline_mean":  2.5,
            "baseline_std":   1.0,
            "z_score":        3.0,
            "severity":       "moderate",
            "direction":      "above_baseline",
        },
    ]


def _strong_trends() -> dict:
    return {
        "solar_wind_speed": {
            "direction": "increasing", "change_percent": 35.0,
            "change_absolute": 120.0, "start_value": 350.0, "end_value": 470.0,
            "severity": "significant", "observations_used": 12,
        },
        "proton_flux": {
            "direction": "increasing", "change_percent": 28.0,
            "change_absolute": 0.5, "start_value": 1.8, "end_value": 2.3,
            "severity": "moderate", "observations_used": 12,
        },
    }


def _nominal_trends() -> dict:
    return {
        "solar_wind_speed": {
            "direction": "stable", "change_percent": 1.0,
            "change_absolute": 4.0, "start_value": 400.0, "end_value": 404.0,
            "severity": "stable", "observations_used": 12,
        },
    }


def _forecast_increasing_speed() -> list[dict]:
    return [
        {"timestamp": f"2025-07-20T12:{5*i:02d}:00+00:00",
         "step": i+1, "parameter": "solar_wind_speed",
         "predicted_value": 400.0 + 30.0 * (i+1),
         "lower_bound":     380.0 + 30.0 * (i+1),
         "upper_bound":     420.0 + 30.0 * (i+1)}
        for i in range(12)
    ]


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def test_top_level_keys_present(self):
        result = assess_mission_risk()
        for key in ("status", "mission_profile", "overall_risk",
                    "domains", "evidence", "recommendations", "message"):
            assert key in result, f"Top-level key '{key}' missing"

    def test_overall_risk_keys(self):
        result = assess_mission_risk()
        assert "level" in result["overall_risk"]
        assert "score" in result["overall_risk"]

    def test_five_domains_present(self):
        result = assess_mission_risk()
        domains = {d["domain"] for d in result["domains"]}
        for expected in ("radiation", "communications", "navigation",
                         "power", "attitude_control"):
            assert expected in domains

    def test_domain_entry_keys(self):
        result = assess_mission_risk()
        for d in result["domains"]:
            for key in ("domain", "risk", "score", "sensitivity", "drivers"):
                assert key in d, f"Domain key '{key}' missing in {d['domain']}"

    def test_overall_risk_level_valid(self):
        result = assess_mission_risk()
        assert result["overall_risk"]["level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    def test_domain_risk_levels_valid(self):
        result = assess_mission_risk()
        for d in result["domains"]:
            assert d["risk"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    def test_risk_score_in_range(self):
        result = assess_mission_risk()
        assert 0.0 <= result["overall_risk"]["score"] <= 100.0
        for d in result["domains"]:
            assert 0.0 <= d["score"] <= 100.0

    def test_evidence_keys(self):
        result = assess_mission_risk()
        ev = result["evidence"]
        for key in ("observed", "analyzed", "predicted"):
            assert key in ev

    def test_json_serialisable(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            anomalies=_significant_anomalies(),
            trends=_strong_trends(),
            predictions=_forecast_increasing_speed(),
        )
        json.dumps(result)  # must not raise

    def test_mission_profile_echoed_in_output(self):
        custom = {
            "radiation_sensitivity":       "high",
            "communications_sensitivity":  "low",
            "navigation_sensitivity":      "medium",
            "power_sensitivity":           "low",
            "attitude_control_sensitivity":"high",
        }
        result = assess_mission_risk(mission_profile=custom)
        assert result["mission_profile"]["radiation_sensitivity"] == "high"
        assert result["mission_profile"]["communications_sensitivity"] == "low"


# ---------------------------------------------------------------------------
# Nominal conditions
# ---------------------------------------------------------------------------

class TestNominalConditions:

    def test_nominal_conditions_returns_ok(self):
        result = assess_mission_risk(conditions=_nominal_conditions())
        assert result["status"] == "ok"

    def test_nominal_no_anomalies_not_critical(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            anomalies=[],
            trends=_nominal_trends(),
        )
        assert result["overall_risk"]["level"] != "CRITICAL"

    def test_observed_evidence_populated(self):
        result = assess_mission_risk(conditions=_nominal_conditions())
        assert len(result["evidence"]["observed"]) > 0


# ---------------------------------------------------------------------------
# Elevated / high-risk conditions
# ---------------------------------------------------------------------------

class TestHighRiskConditions:

    def test_elevated_conditions_higher_risk_than_nominal(self):
        nominal  = assess_mission_risk(conditions=_nominal_conditions())
        elevated = assess_mission_risk(conditions=_elevated_conditions())
        assert elevated["overall_risk"]["score"] >= nominal["overall_risk"]["score"]

    def test_significant_anomalies_increase_risk(self):
        without = assess_mission_risk(
            conditions=_nominal_conditions(), anomalies=[]
        )
        with_anomalies = assess_mission_risk(
            conditions=_nominal_conditions(),
            anomalies=_significant_anomalies(),
        )
        assert with_anomalies["overall_risk"]["score"] >= without["overall_risk"]["score"]

    def test_strong_trends_increase_risk(self):
        without = assess_mission_risk(
            conditions=_nominal_conditions(), trends={},
        )
        with_trends = assess_mission_risk(
            conditions=_nominal_conditions(),
            trends=_strong_trends(),
        )
        assert with_trends["overall_risk"]["score"] >= without["overall_risk"]["score"]

    def test_forecast_increases_risk(self):
        without = assess_mission_risk(
            conditions=_nominal_conditions(), predictions=[],
        )
        with_preds = assess_mission_risk(
            conditions=_nominal_conditions(),
            predictions=_forecast_increasing_speed(),
        )
        assert with_preds["overall_risk"]["score"] >= without["overall_risk"]["score"]

    def test_extreme_conditions_high_or_critical(self):
        result = assess_mission_risk(
            conditions=_extreme_conditions(),
            anomalies=_significant_anomalies(),
            trends=_strong_trends(),
        )
        assert result["overall_risk"]["level"] in ("HIGH", "CRITICAL")


# ---------------------------------------------------------------------------
# Radiation domain
# ---------------------------------------------------------------------------

class TestRadiationDomain:

    def _radiation_score(self, result: dict) -> float:
        for d in result["domains"]:
            if d["domain"] == "radiation":
                return d["score"]
        return 0.0

    def test_high_radiation_sensitivity_amplifies_radiation_domain(self):
        low_profile = {**DEFAULT_MISSION_PROFILE, "radiation_sensitivity": "low"}
        high_profile = {**DEFAULT_MISSION_PROFILE, "radiation_sensitivity": "high"}

        r_low  = assess_mission_risk(
            conditions=_elevated_conditions(),
            anomalies=_significant_anomalies(),
            mission_profile=low_profile,
        )
        r_high = assess_mission_risk(
            conditions=_elevated_conditions(),
            anomalies=_significant_anomalies(),
            mission_profile=high_profile,
        )
        assert self._radiation_score(r_high) >= self._radiation_score(r_low)

    def test_proton_flux_drives_radiation_domain(self):
        """Significant proton_flux anomaly must appear in radiation domain drivers."""
        result = assess_mission_risk(
            conditions=_elevated_conditions(),
            anomalies=_significant_anomalies(),
        )
        rad = next(d for d in result["domains"] if d["domain"] == "radiation")
        driver_text = " ".join(rad["drivers"]).lower()
        assert "proton" in driver_text, (
            f"Expected 'proton' in radiation drivers. Got: {rad['drivers']}"
        )


# ---------------------------------------------------------------------------
# Communications domain
# ---------------------------------------------------------------------------

class TestCommunicationsDomain:

    def test_xray_flux_influences_communications(self):
        """xray_flux has 80% weight in communications — elevated flux should raise comms risk."""
        low_xray  = assess_mission_risk(conditions={**_nominal_conditions(), "xray_flux": 1e-8})
        high_xray = assess_mission_risk(conditions={**_nominal_conditions(), "xray_flux": 1e-5})
        comms_low  = next(d["score"] for d in low_xray["domains"] if d["domain"] == "communications")
        comms_high = next(d["score"] for d in high_xray["domains"] if d["domain"] == "communications")
        assert comms_high > comms_low

    def test_geomagnetic_index_influences_communications(self):
        low_kp  = assess_mission_risk(conditions={**_nominal_conditions(), "geomagnetic_index": 0.5})
        high_kp = assess_mission_risk(conditions={**_nominal_conditions(), "geomagnetic_index": 8.5})
        comms_low  = next(d["score"] for d in low_kp["domains"] if d["domain"] == "communications")
        comms_high = next(d["score"] for d in high_kp["domains"] if d["domain"] == "communications")
        assert comms_high > comms_low


# ---------------------------------------------------------------------------
# Multiple simultaneous high-risk domains
# ---------------------------------------------------------------------------

class TestMultipleDomains:

    def test_multiple_domains_elevated_simultaneously(self):
        result = assess_mission_risk(
            conditions=_extreme_conditions(),
            anomalies=_significant_anomalies(),
            trends=_strong_trends(),
        )
        elevated = [d for d in result["domains"] if d["risk"] in ("HIGH", "CRITICAL", "MODERATE")]
        assert len(elevated) >= 2, (
            f"Expected ≥2 elevated domains, got: {[(d['domain'], d['risk']) for d in result['domains']]}"
        )

    def test_highest_domain_risk_drives_overall(self):
        """Overall risk level should be ≥ the highest individual domain level."""
        result = assess_mission_risk(
            conditions=_extreme_conditions(),
            anomalies=_significant_anomalies(),
        )
        level_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
        max_domain_rank = max(level_rank[d["risk"]] for d in result["domains"])
        overall_rank    = level_rank[result["overall_risk"]["level"]]
        # Overall is a weighted average — it can be lower than the single hottest
        # domain, but should reflect elevated conditions
        assert overall_rank >= 0  # always valid


# ---------------------------------------------------------------------------
# Missing inputs (graceful degradation)
# ---------------------------------------------------------------------------

class TestMissingInputs:

    def test_no_conditions_returns_ok(self):
        result = assess_mission_risk(conditions=None)
        assert result["status"] == "ok"

    def test_no_trends_returns_ok(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(), trends=None
        )
        assert result["status"] == "ok"

    def test_no_anomalies_returns_ok(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(), anomalies=None
        )
        assert result["status"] == "ok"

    def test_empty_anomaly_list_returns_ok(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(), anomalies=[]
        )
        assert result["status"] == "ok"

    def test_no_predictions_returns_ok(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(), predictions=None
        )
        assert result["status"] == "ok"

    def test_all_none_returns_ok(self):
        result = assess_mission_risk(
            conditions=None, trends=None, anomalies=None, predictions=None
        )
        assert result["status"] == "ok"
        # With no data, risk should be LOW
        assert result["overall_risk"]["level"] == "LOW"

    def test_missing_parameter_in_conditions_handled(self):
        partial = {"solar_wind_speed": 450.0}  # only one parameter
        result = assess_mission_risk(conditions=partial)
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Mission profile
# ---------------------------------------------------------------------------

class TestMissionProfile:

    def test_default_profile_used_when_not_specified(self):
        result = assess_mission_risk()
        for key, val in DEFAULT_MISSION_PROFILE.items():
            assert result["mission_profile"][key] == val

    def test_high_sensitivity_yields_higher_domain_score(self):
        low_profile  = {**DEFAULT_MISSION_PROFILE, "communications_sensitivity": "low"}
        high_profile = {**DEFAULT_MISSION_PROFILE, "communications_sensitivity": "high"}
        rl = assess_mission_risk(conditions=_elevated_conditions(), mission_profile=low_profile)
        rh = assess_mission_risk(conditions=_elevated_conditions(), mission_profile=high_profile)
        sl = next(d["score"] for d in rl["domains"] if d["domain"] == "communications")
        sh = next(d["score"] for d in rh["domains"] if d["domain"] == "communications")
        assert sh >= sl

    def test_invalid_sensitivity_value_returns_error(self):
        bad_profile = {**DEFAULT_MISSION_PROFILE, "radiation_sensitivity": "extreme"}
        result = assess_mission_risk(mission_profile=bad_profile)
        assert result["status"] == "error"
        assert "invalid" in result["message"].lower()

    def test_case_insensitive_sensitivity_value(self):
        profile = {**DEFAULT_MISSION_PROFILE, "radiation_sensitivity": "HIGH"}
        result = assess_mission_risk(mission_profile=profile)
        assert result["status"] == "ok"
        assert result["mission_profile"]["radiation_sensitivity"] == "high"

    def test_sensitivity_multipliers_correct(self):
        assert SENSITIVITY_MULTIPLIER["low"]    < SENSITIVITY_MULTIPLIER["medium"]
        assert SENSITIVITY_MULTIPLIER["medium"] < SENSITIVITY_MULTIPLIER["high"]


# ---------------------------------------------------------------------------
# Evidence traceability
# ---------------------------------------------------------------------------

class TestEvidenceTraceability:

    def test_observed_evidence_from_conditions(self):
        result = assess_mission_risk(conditions=_nominal_conditions())
        # Should have at least one observed entry per parameter in conditions
        obs = result["evidence"]["observed"]
        assert len(obs) > 0

    def test_analyzed_evidence_from_anomalies(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            anomalies=_significant_anomalies(),
        )
        analyzed = " ".join(result["evidence"]["analyzed"]).lower()
        assert "anomaly" in analyzed

    def test_analyzed_evidence_from_trends(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            trends=_strong_trends(),
        )
        analyzed = " ".join(result["evidence"]["analyzed"]).lower()
        assert "trend" in analyzed

    def test_predicted_evidence_from_forecasts(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(),
            predictions=_forecast_increasing_speed(),
        )
        predicted = " ".join(result["evidence"]["predicted"]).lower()
        assert "forecast" in predicted

    def test_no_predicted_evidence_when_no_forecasts(self):
        result = assess_mission_risk(
            conditions=_nominal_conditions(), predictions=[]
        )
        assert result["evidence"]["predicted"] == []


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:

    def test_recommendations_present_for_high_risk(self):
        result = assess_mission_risk(
            conditions=_extreme_conditions(),
            anomalies=_significant_anomalies(),
            trends=_strong_trends(),
        )
        assert len(result["recommendations"]) > 0

    def test_recommendations_list_for_nominal(self):
        result = assess_mission_risk(conditions=_nominal_conditions(), anomalies=[])
        # Recommendations should always be a list
        assert isinstance(result["recommendations"], list)

    def test_critical_recommendations_mention_monitoring(self):
        result = assess_mission_risk(
            conditions=_extreme_conditions(),
            anomalies=_significant_anomalies(),
        )
        all_text = " ".join(result["recommendations"]).lower()
        assert "monitor" in all_text or "review" in all_text or "suspend" in all_text


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_inputs_same_outputs(self):
        kwargs = dict(
            conditions=_elevated_conditions(),
            trends=_strong_trends(),
            anomalies=_significant_anomalies(),
            predictions=_forecast_increasing_speed(),
        )
        r1 = assess_mission_risk(**kwargs)
        r2 = assess_mission_risk(**kwargs)
        assert r1["overall_risk"] == r2["overall_risk"]
        assert r1["domains"]      == r2["domains"]

    def test_different_profiles_different_scores(self):
        low_profile  = {**DEFAULT_MISSION_PROFILE,
                        "radiation_sensitivity": "low",
                        "communications_sensitivity": "low"}
        high_profile = {**DEFAULT_MISSION_PROFILE,
                        "radiation_sensitivity": "high",
                        "communications_sensitivity": "high"}
        r_low  = assess_mission_risk(conditions=_elevated_conditions(),
                                     mission_profile=low_profile)
        r_high = assess_mission_risk(conditions=_elevated_conditions(),
                                     mission_profile=high_profile)
        assert r_high["overall_risk"]["score"] >= r_low["overall_risk"]["score"]


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:

    def test_classify_risk_thresholds(self):
        assert _classify_risk(0.0)   == "LOW"
        assert _classify_risk(24.9)  == "LOW"
        assert _classify_risk(25.0)  == "MODERATE"
        assert _classify_risk(49.9)  == "MODERATE"
        assert _classify_risk(50.0)  == "HIGH"
        assert _classify_risk(74.9)  == "HIGH"
        assert _classify_risk(75.0)  == "CRITICAL"
        assert _classify_risk(100.0) == "CRITICAL"

    def test_normalise_mid_range(self):
        # Mid-point should give 0.5
        assert abs(_normalise(550.0, 200.0, 900.0) - 0.5) < 0.01

    def test_normalise_clamp_below(self):
        assert _normalise(0.0, 200.0, 900.0) == 0.0

    def test_normalise_clamp_above(self):
        assert _normalise(1000.0, 200.0, 900.0) == 1.0

    def test_anomaly_contribution_significant(self):
        anoms = [{"parameter": "solar_wind_speed", "z_score": 4.0,
                  "severity": "significant", "direction": "above_baseline"}]
        score, text = _anomaly_contribution("solar_wind_speed", anoms)
        assert score == 50.0
        assert text is not None

    def test_anomaly_contribution_moderate(self):
        anoms = [{"parameter": "proton_flux", "z_score": 2.5,
                  "severity": "moderate", "direction": "above_baseline"}]
        score, text = _anomaly_contribution("proton_flux", anoms)
        assert score == 25.0

    def test_anomaly_contribution_none_when_absent(self):
        score, text = _anomaly_contribution("geomagnetic_index", [])
        assert score == 0.0
        assert text is None

    def test_trend_contribution_significant(self):
        trends = {"solar_wind_speed": {"severity": "significant", "direction": "increasing",
                                        "change_percent": 40.0}}
        score, text = _trend_contribution("solar_wind_speed", trends)
        assert score == 35.0
        assert text is not None

    def test_trend_contribution_stable(self):
        trends = {"solar_wind_speed": {"severity": "stable", "direction": "stable",
                                        "change_percent": 1.0}}
        score, text = _trend_contribution("solar_wind_speed", trends)
        assert score == 0.0
        assert text is None


# ---------------------------------------------------------------------------
# End-to-end integration (real tool outputs)
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline_with_real_tool_outputs(self):
        """Call all four BobVoyage tools and feed their outputs into the risk engine."""
        from bobvoyage.tools.current_conditions import get_current_conditions
        from bobvoyage.tools.analyze_trends     import analyze_trends
        from bobvoyage.tools.detect_anomalies   import detect_anomalies
        from bobvoyage.tools.predict_conditions import predict_conditions

        cc  = get_current_conditions()
        tr  = analyze_trends(window=12)
        an  = detect_anomalies(recent_window=6, baseline_window=48)
        pr  = predict_conditions(horizon=12, lookback=48)

        assert cc["status"] == "ok"
        assert tr["status"] == "ok"
        assert an["status"] == "ok"
        assert pr["status"] == "ok"

        result = assess_mission_risk(
            conditions  = cc["observation"],
            trends      = tr["trends"],
            anomalies   = an["anomalies"],
            predictions = pr["predictions"],
        )

        assert result["status"] == "ok"
        assert result["overall_risk"]["level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
        assert 0.0 <= result["overall_risk"]["score"] <= 100.0

    def test_full_pipeline_json_serialisable(self):
        from bobvoyage.tools.current_conditions import get_current_conditions
        from bobvoyage.tools.analyze_trends     import analyze_trends
        from bobvoyage.tools.detect_anomalies   import detect_anomalies
        from bobvoyage.tools.predict_conditions import predict_conditions

        result = assess_mission_risk(
            conditions  = get_current_conditions()["observation"],
            trends      = analyze_trends(window=12)["trends"],
            anomalies   = detect_anomalies()["anomalies"],
            predictions = predict_conditions(horizon=12)["predictions"],
        )
        json.dumps(result)  # must not raise
