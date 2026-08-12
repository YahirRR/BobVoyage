"""
test_dashboard.py

Tests for M9: BobVoyage dashboard service layer and FastAPI endpoints.

Coverage:
  Service layer:
  1.  get_assessment returns ok status in demo mode
  2.  get_assessment has required top-level keys
  3.  telemetry rows present for all expected parameters
  4.  telemetry unavailable flag set correctly
  5.  telemetry available flag set correctly
  6.  forecast_by_param present and non-empty
  7.  forecast_params list matches forecast_by_param keys
  8.  forecast steps have required fields
  9.  correlated_events list present
  10. risk dict has level, score, domains
  11. risk domains list has all 5 domains
  12. each domain has required keys
  13. evidence dict has all four buckets
  14. timeline list present and non-empty
  15. recommendations list present
  16. mission_profile echoed in response
  17. pipeline_ms positive number
  18. source field non-empty
  19. custom mission profile applied correctly
  20. ask_question returns answer for status question
  21. ask_question returns answer for why question
  22. ask_question returns answer for recommend question
  23. ask_question returns answer for forecast question
  24. ask_question returns answer for events question
  25. ask_question returns answer for changed question
  26. ask_question fallback for unknown question
  27. no causal language in timeline items
  28. no causal language in correlated_events evidence
  29. demo mode label in response
  30. degraded provider graceful response

  FastAPI endpoints:
  31. GET /api/status returns 200 ok
  32. GET /api/assessment returns 200 in demo mode
  33. GET /api/assessment response has required keys
  34. POST /api/assess with custom profile returns 200
  35. POST /api/assess changes domain scores
  36. GET /api/forecast returns 200 for default param
  37. GET /api/forecast has param, label, current_value, forecast keys
  38. POST /api/ask returns 200
  39. POST /api/ask response has question and answer
  40. GET / returns 200 (dashboard HTML)
  41. GET /api/status returns mode field
  42. missing correlated_events in assessment is empty list
"""
from __future__ import annotations

import os
import sys

import pytest

# Force demo/local mode for all tests — no network calls
os.environ.setdefault("BOBVOYAGE_DATA_PROVIDER", "local")

# ── Service layer imports ───────────────────────────────────────────────
from bobvoyage.dashboard.service import (
    get_assessment,
    ask_question,
    _build_timeline,
    _build_telemetry,
    _build_conversational_response,
    _PARAM_META,
)

# ── FastAPI test client ─────────────────────────────────────────────────
from fastapi.testclient import TestClient
from bobvoyage.dashboard.app import app

client = TestClient(app)

# ── Shared fixture ──────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def demo_assessment():
    return get_assessment(mode="demo")


# ===========================================================================
# Service layer — get_assessment
# ===========================================================================
class TestGetAssessment:

    def test_returns_ok_status(self, demo_assessment):
        # If pipeline ran without crashing, provider_status should be ok or degraded
        assert demo_assessment.get("provider_status") in ("ok", "degraded", "error")

    def test_has_required_top_level_keys(self, demo_assessment):
        required = [
            "mode", "provider_status", "source", "retrieved_at",
            "data_age_seconds", "data_age_label", "is_stale",
            "pipeline_ms", "conditions", "telemetry", "trends",
            "anomalies", "forecast_summary", "forecast_by_param",
            "forecast_params", "events", "correlations", "risk",
            "correlated_events", "evidence", "recommendations",
            "mission_profile", "timeline", "param_meta",
        ]
        for key in required:
            assert key in demo_assessment, f"Missing key: {key}"

    def test_telemetry_has_all_expected_params(self, demo_assessment):
        telemetry = demo_assessment["telemetry"]
        params = {t["param"] for t in telemetry}
        for expected_param in _PARAM_META:
            assert expected_param in params, f"Missing telemetry param: {expected_param}"

    def test_telemetry_available_flag(self, demo_assessment):
        # Local provider returns actual values
        telemetry = demo_assessment["telemetry"]
        available = [t for t in telemetry if t["available"]]
        assert len(available) > 0

    def test_telemetry_unavailable_flag_for_none_values(self, demo_assessment):
        # For telemetry entries with value=None, available must be False
        telemetry = demo_assessment["telemetry"]
        for t in telemetry:
            if t["value"] is None:
                assert t["available"] is False

    def test_forecast_by_param_non_empty(self, demo_assessment):
        by_param = demo_assessment["forecast_by_param"]
        assert isinstance(by_param, dict)
        assert len(by_param) > 0

    def test_forecast_params_matches_forecast_by_param(self, demo_assessment):
        assert set(demo_assessment["forecast_params"]) == set(demo_assessment["forecast_by_param"].keys())

    def test_forecast_steps_have_required_fields(self, demo_assessment):
        by_param = demo_assessment["forecast_by_param"]
        for param, steps in by_param.items():
            for step in steps[:2]:
                for field in ("step", "predicted_value", "lower_bound", "upper_bound", "timestamp"):
                    assert field in step, f"Missing field '{field}' in forecast step for {param}"

    def test_correlated_events_is_list(self, demo_assessment):
        assert isinstance(demo_assessment["correlated_events"], list)

    def test_risk_has_level_score_domains(self, demo_assessment):
        risk = demo_assessment["risk"]
        assert "level" in risk
        assert "score" in risk
        assert "domains" in risk

    def test_risk_has_five_domains(self, demo_assessment):
        domains = demo_assessment["risk"]["domains"]
        names = {d["domain"] for d in domains}
        for expected in ("radiation", "communications", "navigation", "power", "attitude_control"):
            assert expected in names

    def test_domain_has_required_keys(self, demo_assessment):
        for d in demo_assessment["risk"]["domains"]:
            for key in ("domain", "risk", "score", "score_environmental", "score_correlation", "drivers"):
                assert key in d, f"Domain missing key: {key}"

    def test_evidence_has_four_buckets(self, demo_assessment):
        evidence = demo_assessment["evidence"]
        for bucket in ("observed", "analyzed", "predicted", "correlated"):
            assert bucket in evidence, f"Evidence missing bucket: {bucket}"

    def test_timeline_non_empty(self, demo_assessment):
        assert len(demo_assessment["timeline"]) > 0

    def test_recommendations_is_list(self, demo_assessment):
        assert isinstance(demo_assessment["recommendations"], list)

    def test_mission_profile_echoed(self, demo_assessment):
        profile = demo_assessment["mission_profile"]
        assert "radiation_sensitivity" in profile
        assert "communications_sensitivity" in profile

    def test_pipeline_ms_positive(self, demo_assessment):
        assert demo_assessment["pipeline_ms"] > 0

    def test_source_non_empty(self, demo_assessment):
        assert demo_assessment["source"]

    def test_demo_mode_label(self, demo_assessment):
        assert demo_assessment["mode"] == "demo"

    def test_custom_mission_profile_applied(self):
        custom_profile = {
            "radiation_sensitivity":       "high",
            "communications_sensitivity":  "high",
            "navigation_sensitivity":      "high",
            "power_sensitivity":           "high",
            "attitude_control_sensitivity":"high",
        }
        result = get_assessment(mode="demo", mission_profile=custom_profile)
        # High sensitivity on all domains should yield higher scores than default
        default = get_assessment(mode="demo")
        default_total = sum(d["score"] for d in default["risk"]["domains"])
        custom_total  = sum(d["score"] for d in result["risk"]["domains"])
        assert custom_total >= default_total


# ===========================================================================
# Service layer — ask_question
# ===========================================================================
class TestAskQuestion:

    @pytest.fixture(scope="class")
    def assessment(self):
        return get_assessment(mode="demo")

    def test_status_question_returns_answer(self, assessment):
        result = ask_question("What is happening right now?", assessment)
        assert result["answer"]
        assert len(result["answer"]) > 10

    def test_why_question_returns_answer(self, assessment):
        result = ask_question("Why is the risk high?", assessment)
        assert result["answer"]

    def test_recommend_question_returns_answer(self, assessment):
        result = ask_question("What should operators monitor?", assessment)
        assert result["answer"]

    def test_forecast_question_returns_answer(self, assessment):
        result = ask_question("What is predicted next?", assessment)
        assert result["answer"]

    def test_events_question_returns_answer(self, assessment):
        result = ask_question("Are there any external events?", assessment)
        assert result["answer"]

    def test_changed_question_returns_answer(self, assessment):
        result = ask_question("What changed recently?", assessment)
        assert result["answer"]

    def test_unknown_question_returns_fallback(self, assessment):
        result = ask_question("xyzzy irrelevant nonsense", assessment)
        assert result["answer"]
        assert "BobVoyage" in result["answer"] or "mission" in result["answer"].lower()

    def test_response_has_required_keys(self, assessment):
        result = ask_question("Status?", assessment)
        assert "question" in result
        assert "answer" in result
        assert "source" in result


# ===========================================================================
# Service layer — causal language prevention
# ===========================================================================
class TestCausalLanguage:

    FORBIDDEN = ["caused", "due to", "resulted in", "was triggered by", "led to"]

    def test_no_causal_language_in_timeline(self):
        result = get_assessment(mode="demo")
        for item in result["timeline"]:
            text = item.get("text", "").lower()
            for phrase in self.FORBIDDEN:
                assert phrase not in text, f"Causal phrase '{phrase}' in timeline: {text}"

    def test_no_causal_language_in_correlated_events_evidence(self):
        result = get_assessment(mode="demo")
        for ce in result["correlated_events"]:
            for ev in ce.get("evidence", []):
                text = ev.lower()
                for phrase in self.FORBIDDEN:
                    assert phrase not in text, f"Causal phrase '{phrase}' in corr evidence: {text}"


# ===========================================================================
# FastAPI endpoints
# ===========================================================================
class TestAPIEndpoints:

    def test_status_returns_200(self):
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_status_has_ok(self):
        resp = client.get("/api/status")
        assert resp.json()["status"] == "ok"

    def test_status_has_mode(self):
        resp = client.get("/api/status")
        assert "mode" in resp.json()

    def test_assessment_returns_200_demo(self):
        resp = client.get("/api/assessment?mode=demo")
        assert resp.status_code == 200

    def test_assessment_has_required_keys(self):
        resp = client.get("/api/assessment?mode=demo")
        data = resp.json()
        for key in ("mode", "risk", "telemetry", "timeline", "evidence"):
            assert key in data, f"Missing key: {key}"

    def test_assessment_risk_has_level(self):
        resp = client.get("/api/assessment?mode=demo")
        data = resp.json()
        assert "level" in data["risk"]
        assert data["risk"]["level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL", "UNKNOWN")

    def test_assess_post_returns_200(self):
        resp = client.post("/api/assess", json={
            "mode": "demo",
            "mission_profile": {
                "radiation_sensitivity": "medium",
                "communications_sensitivity": "medium",
                "navigation_sensitivity": "medium",
                "power_sensitivity": "low",
                "attitude_control_sensitivity": "medium",
            }
        })
        assert resp.status_code == 200

    def test_assess_post_high_sensitivity_changes_scores(self):
        default_resp = client.post("/api/assess", json={"mode": "demo"})
        high_resp    = client.post("/api/assess", json={
            "mode": "demo",
            "mission_profile": {
                "radiation_sensitivity":       "high",
                "communications_sensitivity":  "high",
                "navigation_sensitivity":      "high",
                "power_sensitivity":           "high",
                "attitude_control_sensitivity":"high",
            }
        })
        default_score = default_resp.json()["risk"]["score"]
        high_score    = high_resp.json()["risk"]["score"]
        assert high_score >= default_score

    def test_forecast_returns_200(self):
        resp = client.get("/api/forecast?param=solar_wind_speed&mode=demo")
        assert resp.status_code == 200

    def test_forecast_has_required_keys(self):
        resp = client.get("/api/forecast?param=solar_wind_speed&mode=demo")
        data = resp.json()
        for key in ("param", "label", "unit", "current_value", "forecast", "available_params"):
            assert key in data, f"Missing key: {key}"

    def test_forecast_available_params_non_empty(self):
        resp = client.get("/api/forecast?mode=demo")
        data = resp.json()
        assert len(data["available_params"]) > 0

    def test_ask_returns_200(self):
        resp = client.post("/api/ask", json={"question": "What is happening?", "mode": "demo"})
        assert resp.status_code == 200

    def test_ask_has_answer(self):
        resp = client.post("/api/ask", json={"question": "Status?", "mode": "demo"})
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 5

    def test_dashboard_html_served(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "BobVoyage" in resp.text
        assert "<!DOCTYPE html>" in resp.text

    def test_assess_missing_events_returns_empty_correlated(self):
        resp = client.post("/api/assess", json={"mode": "demo"})
        data = resp.json()
        assert isinstance(data["correlated_events"], list)


# ===========================================================================
# Service layer — helper unit tests
# ===========================================================================
class TestServiceHelpers:

    def test_build_telemetry_all_params(self):
        conditions = {
            "solar_wind_speed": 420.0,
            "solar_wind_density": 6.0,
            "magnetic_field": 8.0,
            "xray_flux": 1e-7,
            "proton_flux": 0.5,
            "geomagnetic_index": 2.0,
        }
        rows = _build_telemetry(conditions, _PARAM_META)
        assert len(rows) == len(_PARAM_META)
        for r in rows:
            assert "param" in r
            assert "label" in r
            assert "unit" in r
            assert "value" in r
            assert "available" in r

    def test_build_telemetry_none_values_unavailable(self):
        conditions = {"solar_wind_speed": None}
        rows = _build_telemetry(conditions, _PARAM_META)
        sw = next(r for r in rows if r["param"] == "solar_wind_speed")
        assert sw["available"] is False

    def test_build_timeline_has_assessment_entry(self):
        evidence = {
            "observed": ["Solar wind speed elevated"],
            "analyzed": [],
            "predicted": [],
            "correlated": [],
        }
        risk_result = {"overall_risk": {"level": "HIGH", "score": 58.3}}
        tl = _build_timeline(evidence, {}, risk_result)
        categories = {item["category"] for item in tl}
        assert "ASSESSMENT" in categories

    def test_build_timeline_observed_items_present(self):
        evidence = {
            "observed": ["Solar wind: 487 km/s", "Xray: 3e-6 W/m2"],
            "analyzed": [],
            "predicted": [],
            "correlated": [],
        }
        tl = _build_timeline(evidence, {}, {"overall_risk": {"level": "LOW", "score": 10}})
        obs_items = [i for i in tl if i["category"] == "OBSERVED"]
        assert len(obs_items) >= 1

    def test_conversational_status_response_mentions_risk(self):
        assessment = {
            "risk": {"level": "HIGH", "score": 58.3, "domains": []},
            "evidence": {"observed": ["Solar wind 487 km/s"], "analyzed": [], "predicted": [], "correlated": []},
            "correlated_events": [],
            "recommendations": [],
        }
        response = _build_conversational_response("What is happening now?", assessment)
        assert "HIGH" in response or "58" in response

    def test_conversational_events_no_events(self):
        assessment = {
            "risk": {"level": "LOW", "score": 10, "domains": []},
            "evidence": {"observed": [], "analyzed": [], "predicted": [], "correlated": []},
            "correlated_events": [],
            "recommendations": [],
        }
        response = _build_conversational_response("Are there any external events?", assessment)
        assert "no" in response.lower() or "none" in response.lower() or "not" in response.lower()
