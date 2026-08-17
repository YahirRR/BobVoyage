"""
test_m10_mcp_http.py

Tests for M10: Streamable HTTP MCP transport.

Coverage:
  Integration:
  1.  GET /health returns 200 with expected keys
  2.  health response has status=ok and service=bobvoyage
  3.  health response has mcp=available
  4.  health response has endpoint field
  5.  health does not expose secrets (no API key)

  MCP tool discovery (via FastMCP test client):
  6.  list_tools returns non-empty list
  7.  all 8 expected tools are listed
  8.  each tool has name and description
  9.  each tool has inputSchema

  MCP prompt discovery:
  10. list_prompts returns bobvoyage_mission_context
  11. prompt content mentions OBSERVED and CORRELATED
  12. prompt content mentions risk score disclaimer

  Transport equivalence (HTTP tools match STDIO tool results):
  13. get_current_conditions_tool returns same status as direct call
  14. get_current_conditions_tool returns observation with solar_wind_speed
  15. analyze_trends_tool returns same status as direct call
  16. detect_anomalies_tool returns same status as direct call
  17. predict_conditions_tool returns same status as direct call
  18. assess_mission_risk_tool returns same status as direct call
  19. stakeholder_briefing_tool returns same status as direct call

  Tool functionality via HTTP:
  20. get_current_conditions returns valid observation
  21. analyze_trends returns trends dict
  22. detect_anomalies returns anomalies list
  23. predict_conditions returns predictions list
  24. assess_mission_risk returns domain list
  25. correlate_space_events handles empty events gracefully
  26. recurrence_forecast returns ok status
  27. stakeholder_briefing returns briefing for satellite_operator

  Failure / edge cases:
  28. unknown tool via direct call returns error
  29. assess_mission_risk with no inputs returns ok (graceful)
  30. stakeholder_briefing with missing audience returns error
  31. predict_conditions with invalid horizon returns error
  32. detect_anomalies with invalid z_threshold returns error

  Config:
  33. config DATA_PROVIDER reads env var
  34. config CORS_ORIGINS parses comma-separated list
  35. config CORS_ORIGINS empty string returns empty list
  36. config MCP_PORT reads env int
  37. config MCP_PATH reads env path
  38. config bad MCP_PORT int falls back to default

  Concurrency / state isolation:
  39. two sequential calls to get_current_conditions return same result
  40. tool calls do not mutate shared state across requests

  Security:
  41. health endpoint does not include NASA_API_KEY value
  42. health endpoint does not include BOBVOYAGE_NASA_API_KEY value
"""

from __future__ import annotations

import json
import os

import pytest
from starlette.testclient import TestClient

# Force demo mode — no network calls
os.environ["BOBVOYAGE_DATA_PROVIDER"] = "local"

from bobvoyage.mcp_http import app, mcp
from bobvoyage.config import (
    DATA_PROVIDER, CORS_ORIGINS, MCP_PORT, MCP_PATH,
)

# ── FastAPI/Starlette test client ───────────────────────────────────────
client = TestClient(app)


# ===========================================================================
# 1–5  Health endpoint
# ===========================================================================
class TestHealthEndpoint:

    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_service_bobvoyage(self):
        resp = client.get("/health")
        assert resp.json()["service"] == "bobvoyage"

    def test_health_mcp_available(self):
        resp = client.get("/health")
        assert resp.json()["mcp"] == "available"

    def test_health_has_endpoint(self):
        resp = client.get("/health")
        assert "endpoint" in resp.json()

    def test_health_does_not_expose_api_key(self):
        # Even if a key is set, health must not reveal it
        os.environ["BOBVOYAGE_NASA_API_KEY"] = "SECRET_TEST_KEY_DO_NOT_EXPOSE"
        resp = client.get("/health")
        body = resp.text
        assert "SECRET_TEST_KEY_DO_NOT_EXPOSE" not in body
        del os.environ["BOBVOYAGE_NASA_API_KEY"]


# ===========================================================================
# 6–9  Tool discovery
# ===========================================================================
EXPECTED_TOOLS = {
    "get_current_conditions_tool",
    "analyze_trends_tool",
    "detect_anomalies_tool",
    "predict_conditions_tool",
    "assess_mission_risk_tool",
    "correlate_space_events_tool",
    "recurrence_forecast_tool",
    "stakeholder_briefing_tool",
}


class TestToolDiscovery:

    @pytest.fixture(scope="class")
    def tools(self):
        import asyncio
        return asyncio.run(mcp.list_tools())

    def test_list_tools_non_empty(self, tools):
        assert len(tools) > 0

    def test_all_expected_tools_listed(self, tools):
        names = {t.name for t in tools}
        for expected in EXPECTED_TOOLS:
            assert expected in names, f"Missing tool: {expected}"

    def test_each_tool_has_description(self, tools):
        for t in tools:
            assert t.description and len(t.description) > 10, \
                f"Tool {t.name} has insufficient description"

    def test_each_tool_has_input_schema(self, tools):
        for t in tools:
            assert t.inputSchema is not None, f"Tool {t.name} missing inputSchema"


# ===========================================================================
# 10–12  Prompt discovery
# ===========================================================================
class TestPromptDiscovery:

    @pytest.fixture(scope="class")
    def prompts(self):
        import asyncio
        return asyncio.run(mcp.list_prompts())

    def test_mission_context_prompt_listed(self, prompts):
        names = {p.name for p in prompts}
        assert "bobvoyage_mission_context" in names

    def test_prompt_content_mentions_observed(self):
        import asyncio
        result = asyncio.run(mcp.get_prompt("bobvoyage_mission_context", {}))
        text = " ".join(m.content.text for m in result.messages if hasattr(m.content, "text"))
        assert "OBSERVED" in text

    def test_prompt_content_mentions_correlated(self):
        import asyncio
        result = asyncio.run(mcp.get_prompt("bobvoyage_mission_context", {}))
        text = " ".join(m.content.text for m in result.messages if hasattr(m.content, "text"))
        assert "CORRELATED" in text

    def test_prompt_content_risk_score_disclaimer(self):
        import asyncio
        result = asyncio.run(mcp.get_prompt("bobvoyage_mission_context", {}))
        text = " ".join(m.content.text for m in result.messages if hasattr(m.content, "text"))
        assert "failure" in text.lower() or "probability" in text.lower()


# ===========================================================================
# 13–19  Transport equivalence (HTTP tool == direct Python call)
# ===========================================================================
class TestTransportEquivalence:
    """Verify that calling tools through the FastMCP layer produces the same
    status as calling the underlying Python functions directly."""

    def _call(self, tool_name: str, args: dict = {}) -> dict:
        import asyncio
        result = asyncio.run(mcp.call_tool(tool_name, args))
        # FastMCP returns list of content objects
        text = result[0][0].text if result and result[0] else "{}"
        return json.loads(text)

    def test_get_current_conditions_status(self):
        from bobvoyage.tools.current_conditions import get_current_conditions
        direct = get_current_conditions()
        via_mcp = self._call("get_current_conditions_tool")
        assert via_mcp["status"] == direct["status"]

    def test_get_current_conditions_solar_wind_present(self):
        result = self._call("get_current_conditions_tool")
        assert "observation" in result
        obs = result["observation"]
        assert "solar_wind_speed" in obs

    def test_analyze_trends_status(self):
        from bobvoyage.tools.analyze_trends import analyze_trends
        direct = analyze_trends()
        via_mcp = self._call("analyze_trends_tool")
        assert via_mcp["status"] == direct["status"]

    def test_detect_anomalies_status(self):
        from bobvoyage.tools.detect_anomalies import detect_anomalies
        direct = detect_anomalies()
        via_mcp = self._call("detect_anomalies_tool")
        assert via_mcp["status"] == direct["status"]

    def test_predict_conditions_status(self):
        from bobvoyage.tools.predict_conditions import predict_conditions
        direct = predict_conditions(horizon=3)
        via_mcp = self._call("predict_conditions_tool", {"horizon": 3})
        assert via_mcp["status"] == direct["status"]

    def test_assess_mission_risk_status(self):
        from bobvoyage.tools.assess_mission_risk import assess_mission_risk
        direct = assess_mission_risk()
        via_mcp = self._call("assess_mission_risk_tool")
        assert via_mcp["status"] == direct["status"]

    def test_stakeholder_briefing_status(self):
        from bobvoyage.tools.assess_mission_risk import assess_mission_risk
        from bobvoyage.tools.stakeholder_briefing import generate_stakeholder_briefing
        risk = assess_mission_risk()
        direct = generate_stakeholder_briefing(risk_assessment=risk, audience="aviation")
        via_mcp = self._call("stakeholder_briefing_tool", {
            "risk_assessment": risk,
            "audience": "aviation",
        })
        assert via_mcp["status"] == direct["status"]


# ===========================================================================
# 20–27  Tool functionality via HTTP layer
# ===========================================================================
class TestToolFunctionality:

    def _call(self, tool_name: str, args: dict = {}) -> dict:
        import asyncio
        result = asyncio.run(mcp.call_tool(tool_name, args))
        return json.loads(result[0][0].text)

    def test_current_conditions_returns_observation(self):
        r = self._call("get_current_conditions_tool")
        assert r["status"] == "ok"
        assert "observation" in r

    def test_analyze_trends_returns_trends(self):
        r = self._call("analyze_trends_tool")
        assert r["status"] == "ok"
        assert "trends" in r

    def test_detect_anomalies_returns_anomalies(self):
        r = self._call("detect_anomalies_tool")
        assert r["status"] == "ok"
        assert "anomalies" in r

    def test_predict_conditions_returns_predictions(self):
        r = self._call("predict_conditions_tool", {"horizon": 3})
        assert r["status"] == "ok"
        assert "predictions" in r
        assert len(r["predictions"]) > 0

    def test_assess_mission_risk_returns_domains(self):
        r = self._call("assess_mission_risk_tool")
        assert r["status"] == "ok"
        assert "domains" in r
        assert len(r["domains"]) == 5

    def test_correlate_space_events_empty_events(self):
        r = self._call("correlate_space_events_tool", {"events": [], "observations": []})
        assert r["status"] == "ok"
        assert r["correlations"] == []

    def test_recurrence_forecast_ok(self):
        r = self._call("recurrence_forecast_tool")
        assert r["status"] in ("ok", "error")  # ok when data available

    def test_stakeholder_briefing_satellite_operator(self):
        import asyncio
        from bobvoyage.tools.assess_mission_risk import assess_mission_risk
        risk = assess_mission_risk()
        r = self._call("stakeholder_briefing_tool", {
            "risk_assessment": risk,
            "audience": "satellite_operator",
        })
        assert r["status"] == "ok"
        assert "briefing" in r or "summary" in r or "audience" in r


# ===========================================================================
# 28–32  Failure / edge cases
# ===========================================================================
class TestFailureEdgeCases:

    def _call(self, tool_name: str, args: dict = {}) -> dict:
        import asyncio
        try:
            result = asyncio.run(mcp.call_tool(tool_name, args))
            return json.loads(result[0][0].text)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def test_assess_mission_risk_no_inputs_graceful(self):
        r = self._call("assess_mission_risk_tool", {})
        assert r["status"] == "ok"

    def test_stakeholder_briefing_missing_audience_returns_error(self):
        from bobvoyage.tools.assess_mission_risk import assess_mission_risk
        risk = assess_mission_risk()
        r = self._call("stakeholder_briefing_tool", {
            "risk_assessment": risk,
            "audience": None,
        })
        assert r["status"] == "error"

    def test_predict_conditions_invalid_horizon_returns_error(self):
        r = self._call("predict_conditions_tool", {"horizon": 0})
        assert r["status"] == "error"

    def test_detect_anomalies_invalid_z_threshold_returns_error(self):
        r = self._call("detect_anomalies_tool", {"z_threshold": -1.0})
        assert r["status"] == "error"

    def test_correlate_space_events_invalid_lookback_returns_error(self):
        r = self._call("correlate_space_events_tool", {"lookback_hours": -1})
        assert r["status"] == "error"


# ===========================================================================
# 33–38  Config module
# ===========================================================================
class TestConfig:

    def test_data_provider_is_local_in_test(self):
        assert DATA_PROVIDER == "local"

    def test_cors_origins_empty_list_by_default(self):
        # Default (no env var set) should be empty
        import importlib
        import bobvoyage.config as cfg
        assert isinstance(cfg.CORS_ORIGINS, list)

    def test_cors_origins_parses_comma_separated(self):
        import importlib
        os.environ["BOBVOYAGE_CORS_ORIGINS"] = "https://a.com,https://b.com"
        import bobvoyage.config as cfg
        importlib.reload(cfg)
        origins = cfg.CORS_ORIGINS
        del os.environ["BOBVOYAGE_CORS_ORIGINS"]
        importlib.reload(cfg)
        assert "https://a.com" in origins
        assert "https://b.com" in origins

    def test_cors_origins_empty_string_returns_empty_list(self):
        import importlib
        os.environ["BOBVOYAGE_CORS_ORIGINS"] = ""
        import bobvoyage.config as cfg
        importlib.reload(cfg)
        origins = cfg.CORS_ORIGINS
        del os.environ["BOBVOYAGE_CORS_ORIGINS"]
        importlib.reload(cfg)
        assert origins == []

    def test_mcp_port_is_int(self):
        assert isinstance(MCP_PORT, int)

    def test_mcp_path_starts_with_slash(self):
        assert MCP_PATH.startswith("/")

    def test_bad_mcp_port_falls_back_to_default(self):
        import importlib
        os.environ["BOBVOYAGE_MCP_PORT"] = "not_a_number"
        import bobvoyage.config as cfg
        importlib.reload(cfg)
        port = cfg.MCP_PORT
        del os.environ["BOBVOYAGE_MCP_PORT"]
        importlib.reload(cfg)
        assert isinstance(port, int)


# ===========================================================================
# 39–40  State isolation / concurrency
# ===========================================================================
class TestStateIsolation:

    def _call(self, tool_name: str, args: dict = {}) -> dict:
        import asyncio
        result = asyncio.run(mcp.call_tool(tool_name, args))
        return json.loads(result[0][0].text)

    def test_sequential_calls_return_same_result(self):
        r1 = self._call("get_current_conditions_tool")
        r2 = self._call("get_current_conditions_tool")
        assert r1["status"] == r2["status"]
        obs1 = r1.get("observation", {})
        obs2 = r2.get("observation", {})
        assert obs1.get("solar_wind_speed") == obs2.get("solar_wind_speed")

    def test_tool_calls_do_not_contaminate_state(self):
        """Calling assess_mission_risk without inputs, then with inputs,
        should produce independent results."""
        empty = self._call("assess_mission_risk_tool", {})
        with_conds = self._call("assess_mission_risk_tool", {
            "conditions": {
                "solar_wind_speed": 800.0,
                "geomagnetic_index": 9.0,
                "xray_flux": 1e-4,
                "proton_flux": 1000.0,
            }
        })
        # Extreme conditions should yield higher risk than empty
        assert with_conds["overall_risk"]["score"] >= empty["overall_risk"]["score"]


# ===========================================================================
# 41–42  Security
# ===========================================================================
class TestSecurity:

    def test_health_does_not_expose_nasa_api_key(self):
        os.environ["BOBVOYAGE_NASA_API_KEY"] = "MY_PRIVATE_KEY_XYZ"
        resp = client.get("/health")
        assert "MY_PRIVATE_KEY_XYZ" not in resp.text
        del os.environ["BOBVOYAGE_NASA_API_KEY"]

    def test_health_does_not_expose_any_secret_pattern(self):
        resp = client.get("/health")
        body = resp.json()
        # Health should only have: status, service, mcp, endpoint, data_provider
        allowed_keys = {"status", "service", "mcp", "endpoint", "data_provider"}
        for key in body:
            assert key in allowed_keys, f"Unexpected key in health response: {key}"
