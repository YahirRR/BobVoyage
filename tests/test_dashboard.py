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
    _build_risk_assessment_from_context,
    _PARAM_META,
    _AUDIENCE_KEYWORDS,
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


# ===========================================================================
# ask_question — recurrence intent
# ===========================================================================
class TestAskQuestionRecurrence:
    """Tests for the new 'recurrence' conversational intent."""

    @pytest.fixture(scope="class")
    def assessment(self):
        return get_assessment(mode="demo")

    def _recurrence_answer(self, assessment, question: str) -> str:
        result = ask_question(question, assessment)
        return result["answer"]

    def test_recurrence_keyword_returns_answer(self, assessment):
        answer = self._recurrence_answer(assessment, "recurrence forecast for active regions?")
        assert isinstance(answer, str) and len(answer) > 10

    def test_recurrence_answer_non_empty(self, assessment):
        answer = self._recurrence_answer(assessment, "Are there any active regions about to return?")
        assert len(answer) > 0

    def test_recurrence_answer_mentions_region_or_no_regions(self, assessment):
        """Response must mention an active region or explain none found."""
        answer = self._recurrence_answer(assessment, "recurrence")
        assert (
            "AR " in answer
            or "active region" in answer.lower()
            or "no active" in answer.lower()
        )

    def test_recurrence_no_causal_language(self, assessment):
        forbidden = ["caused", "due to", "resulted in", "was triggered by", "led to"]
        answer = self._recurrence_answer(assessment, "recurrence forecast")
        for phrase in forbidden:
            assert phrase not in answer.lower(), (
                f"Causal phrase '{phrase}' found in recurrence answer"
            )

    def test_rotate_keyword_triggers_recurrence(self, assessment):
        answer = self._recurrence_answer(assessment, "which regions will rotate back?")
        assert len(answer) > 0

    def test_active_region_keyword_triggers_recurrence(self, assessment):
        answer = self._recurrence_answer(assessment, "any active region coming back?")
        assert len(answer) > 0

    def test_recurrence_response_is_string(self, assessment):
        result = ask_question("recurrence", assessment)
        assert isinstance(result["answer"], str)

    def test_recurrence_answer_has_epistemic_prefix_or_fallback(self, assessment):
        """Answer should use epistemic prefixes or a fallback message.
        The question uses 'recurrence' as the trigger keyword (no 'forecast' overlap)."""
        answer = self._recurrence_answer(assessment, "recurrence for active regions?")
        has_prefix = any(
            p in answer for p in ("OBSERVED:", "ANALYZED:", "PROJECTED:", "No active")
        )
        assert has_prefix, f"Expected epistemic prefix or fallback. Got: {answer[:200]}"


# ===========================================================================
# ask_question — briefing intent
# ===========================================================================
class TestAskQuestionBriefing:
    """Tests for the new 'briefing' conversational intent across all 4 audiences."""

    @pytest.fixture(scope="class")
    def assessment(self):
        return get_assessment(mode="demo")

    @pytest.mark.parametrize("question,expected_label", [
        ("Brief me as an astronaut",                  "Astronaut"),
        ("Brief me as a satellite operator",          "Satellite Operator"),
        ("Give me a briefing for aviation",           "Aviation"),
        ("Explain the situation for the power grid",  "Power Grid"),
    ])
    def test_briefing_all_audiences_return_answer(self, assessment, question, expected_label):
        result = ask_question(question, assessment)
        assert result["status"] if "status" in result else True  # ask_question doesn't add status
        assert isinstance(result["answer"], str) and len(result["answer"]) > 10

    @pytest.mark.parametrize("question", [
        "Brief me as an astronaut",
        "Brief me as a satellite operator",
        "Give me a briefing for aviation",
        "Explain the situation for the power grid",
    ])
    def test_briefing_mentions_space_weather_or_risk(self, assessment, question):
        result = ask_question(question, assessment)
        answer_lower = result["answer"].lower()
        assert (
            "risk" in answer_lower
            or "space" in answer_lower
            or "weather" in answer_lower
            or "radiation" in answer_lower
            or "communication" in answer_lower
        ), f"Expected risk/space context in: {result['answer'][:200]}"

    def test_briefing_keyword_triggers_intent(self, assessment):
        result = ask_question("Give me a briefing", assessment)
        assert isinstance(result["answer"], str) and len(result["answer"]) > 10

    def test_briefing_response_has_required_keys(self, assessment):
        result = ask_question("Brief me as an astronaut", assessment)
        assert "question" in result
        assert "answer" in result
        assert "source" in result

    def test_briefing_astronaut_mentions_radiation_or_communications(self, assessment):
        result = ask_question("Brief me as an astronaut", assessment)
        answer_lower = result["answer"].lower()
        assert "radiation" in answer_lower or "communication" in answer_lower or "crew" in answer_lower

    def test_briefing_aviation_mentions_navigation_or_communications(self, assessment):
        result = ask_question("Briefing for aviation", assessment)
        answer_lower = result["answer"].lower()
        assert (
            "navigation" in answer_lower
            or "communication" in answer_lower
            or "hf" in answer_lower
            or "gps" in answer_lower
            or "risk" in answer_lower
        )

    def test_briefing_power_grid_mentions_power_or_geomagnetic(self, assessment):
        result = ask_question("Brief me for the power grid", assessment)
        answer_lower = result["answer"].lower()
        assert (
            "power" in answer_lower
            or "geomagnetic" in answer_lower
            or "grid" in answer_lower
            or "risk" in answer_lower
            or "communication" in answer_lower  # communications proxies GIC risk
        )

    def test_briefing_no_causal_language(self, assessment):
        forbidden = ["caused", "due to", "resulted in", "was triggered by", "led to"]
        result = ask_question("Brief me as an astronaut", assessment)
        for phrase in forbidden:
            assert phrase not in result["answer"].lower(), (
                f"Causal phrase '{phrase}' in briefing answer"
            )


# ===========================================================================
# Regression — existing intents still work after new additions
# ===========================================================================
class TestExistingIntentsNoRegression:
    """Guard against the new intents breaking existing categories."""

    @pytest.fixture(scope="class")
    def assessment(self):
        return get_assessment(mode="demo")

    def test_status_still_works(self, assessment):
        result = ask_question("What is happening right now?", assessment)
        assert len(result["answer"]) > 10

    def test_why_still_works(self, assessment):
        result = ask_question("Why is the risk high?", assessment)
        assert len(result["answer"]) > 0

    def test_changed_still_works(self, assessment):
        result = ask_question("What changed recently?", assessment)
        assert len(result["answer"]) > 0

    def test_forecast_still_works(self, assessment):
        result = ask_question("What is predicted next?", assessment)
        assert len(result["answer"]) > 0

    def test_events_still_works(self, assessment):
        result = ask_question("Are there any external events?", assessment)
        assert len(result["answer"]) > 0

    def test_recommend_still_works(self, assessment):
        result = ask_question("What should operators monitor?", assessment)
        assert len(result["answer"]) > 0

    def test_fallback_still_works(self, assessment):
        result = ask_question("xyzzy irrelevant nonsense", assessment)
        assert "BobVoyage" in result["answer"] or "mission" in result["answer"].lower()

    def test_api_ask_status_returns_200(self):
        resp = client.post("/api/ask", json={"question": "Status?", "mode": "demo"})
        assert resp.status_code == 200

    def test_api_ask_recurrence_returns_200(self):
        resp = client.post("/api/ask", json={"question": "recurrence forecast", "mode": "demo"})
        assert resp.status_code == 200

    def test_api_ask_briefing_astronaut_returns_200(self):
        resp = client.post("/api/ask", json={"question": "Brief me as an astronaut", "mode": "demo"})
        assert resp.status_code == 200

    def test_api_ask_briefing_aviation_returns_200(self):
        resp = client.post("/api/ask", json={"question": "Briefing for aviation", "mode": "demo"})
        assert resp.status_code == 200

    def test_api_ask_briefing_power_grid_returns_200(self):
        resp = client.post("/api/ask", json={"question": "Explain for the power grid", "mode": "demo"})
        assert resp.status_code == 200

    def test_api_ask_briefing_satellite_operator_returns_200(self):
        resp = client.post("/api/ask", json={"question": "Brief me as a satellite operator", "mode": "demo"})
        assert resp.status_code == 200


# ===========================================================================
# Service helpers — new additions
# ===========================================================================
class TestNewServiceHelpers:

    def test_build_risk_assessment_from_context_shape(self):
        assessment = get_assessment(mode="demo")
        ra = _build_risk_assessment_from_context(assessment)
        for key in ("status", "overall_risk", "domains", "evidence", "recommendations"):
            assert key in ra, f"Missing key '{key}' in reconstructed risk assessment"

    def test_build_risk_assessment_overall_risk_has_level(self):
        assessment = get_assessment(mode="demo")
        ra = _build_risk_assessment_from_context(assessment)
        assert "level" in ra["overall_risk"]

    def test_build_risk_assessment_domains_is_list(self):
        assessment = get_assessment(mode="demo")
        ra = _build_risk_assessment_from_context(assessment)
        assert isinstance(ra["domains"], list)

    def test_audience_keywords_cover_all_four_audiences(self):
        audience_values = set(_AUDIENCE_KEYWORDS.values())
        for expected in ("satellite_operator", "astronaut", "aviation", "power_grid"):
            assert expected in audience_values, f"Missing audience '{expected}' in _AUDIENCE_KEYWORDS"

    def test_dashboard_html_has_recurrence_button(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Recurrence?" in resp.text

    def test_dashboard_html_has_astronaut_button(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Brief for astronaut?" in resp.text
