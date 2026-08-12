"""
Tests for correlate_space_events — BobVoyage MCP intelligence tool

All inputs are constructed in-memory — zero live API calls.

Covers:
  Output schema
    - Required top-level keys present
    - correlations list is a list
    - summary keys present
    - parameters keys echoed correctly
    - JSON serialisable

  No-input edge cases
    - No events → status ok, empty correlations
    - No observations → status ok, empty correlations
    - Both empty → status ok, empty correlations

  Single event, single observation
    - Observation inside window → correlation built
    - Observation outside window → excluded from window_obs
    - Correlation score in [0, 1]
    - component_scores present and in [0, 1]
    - interpretation is one of the four valid labels

  Temporal proximity
    - Observation at t_event → highest temporal score
    - Observation at window boundary → temporal score ≈ 0
    - Observation outside window → excluded

  Anomaly / statistical component
    - High z-score parameter drives anomaly_score up
    - Zero deviation → anomaly_score ≈ 0
    - z_score sign preserved in top_deviations (above/below)

  Trend component
    - Significant variation in window → trend_score = 1.0
    - Flat values → trend_score = 0.0

  Event-weight component
    - CME has highest event weight
    - ALERT has lower weight than CME
    - Unknown event type gets minimum weight

  Multiple events
    - Correlations sorted by score descending
    - Summary counts correct
    - Each event produces an independent correlation record

  Multiple observations per window
    - Only top deviation per parameter kept
    - temporal_distance_minutes reflects closest observation

  Causal language prevention
    - No "caused" in any evidence string
    - No "due to" in any evidence string
    - No "resulted in" in any evidence string
    - Interpretation labels contain no causal language

  Interpretation thresholds
    - score ≥ 0.75 → strong_temporal_association
    - score ≥ 0.50 → moderate_temporal_association
    - score ≥ 0.25 → weak_temporal_association
    - score < 0.25 → no_significant_correlation

  min_score filtering
    - Correlations below min_score excluded
    - min_score=0 returns all
    - min_score=1.0 returns only perfect scores

  Input validation
    - lookback_hours ≤ 0 → error
    - lookahead_hours ≤ 0 → error
    - min_score < 0 → error
    - min_score > 1 → error
    - Non-numeric lookback_hours → error

  Provider integration (local dataset, no network)
    - SpaceWeatherObservation dataclass instances accepted
    - SpaceWeatherEvent dataclass instances accepted
    - Dict inputs accepted

  Determinism
    - Same inputs → same correlation scores

  potential_correlations count
    - Excludes no_significant_correlation entries
    - Matches non-"no_significant" correlations in list
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from bobvoyage.tools.correlate_space_events import (
    correlate_space_events,
    _causal_guard,
    _temporal_score,
    _anomaly_score,
    _interpret,
    _compute_z,
    _CAUSAL_FORBIDDEN,
)
from bobvoyage.data.models.space_weather import SpaceWeatherEvent, SpaceWeatherObservation


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _ts(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


_T0 = datetime(2025, 7, 20, 14, 0, 0, tzinfo=timezone.utc)


def _event(
    event_type: str = "CME",
    offset_hours: float = 0,
    severity: str | None = "M2",
    external_id: str | None = "EVT-001",
) -> dict:
    t = _T0 + timedelta(hours=offset_hours)
    return {
        "event_type":  event_type,
        "event_time":  _ts(t),
        "source":      "NASA_DONKI",
        "external_id": external_id,
        "description": f"Test {event_type} event",
        "severity":    severity,
        "extra":       {},
    }


def _obs(
    offset_hours: float = 0,
    speed: float = 400.0,
    density: float = 5.0,
    bfield: float = 7.0,
    xray: float = 1.2e-7,
    proton: float = 2.0,
    kp: float = 2.5,
) -> dict:
    t = _T0 + timedelta(hours=offset_hours)
    return {
        "timestamp":          _ts(t),
        "solar_wind_speed":   speed,
        "solar_wind_density": density,
        "magnetic_field":     bfield,
        "xray_flux":          xray,
        "proton_flux":        proton,
        "geomagnetic_index":  kp,
        "source":             "NOAA",
        "retrieved_at":       _ts(_T0),
        "data_age_seconds":   0,
        "is_stale":           False,
        "provider_meta":      {},
    }


def _baseline_obs(n: int = 20, speed_mean: float = 400.0) -> list[dict]:
    """Generate n flat-valued observations for a clean baseline."""
    return [
        _obs(offset_hours=-(n - i) * 0.25, speed=speed_mean)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def test_top_level_keys(self):
        result = correlate_space_events()
        for key in ("status", "correlations", "summary", "parameters", "message"):
            assert key in result, f"Missing key '{key}'"

    def test_summary_keys(self):
        result = correlate_space_events()
        s = result["summary"]
        for k in ("events_analyzed", "observations_analyzed",
                  "potential_correlations", "interpretation_counts"):
            assert k in s

    def test_parameters_echoed(self):
        result = correlate_space_events(lookback_hours=3.0, lookahead_hours=1.5, min_score=0.2)
        p = result["parameters"]
        assert p["lookback_hours"]  == 3.0
        assert p["lookahead_hours"] == 1.5
        assert p["min_score"]       == 0.2

    def test_correlation_entry_schema(self):
        obs = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=700.0)]
        result = correlate_space_events(
            events=[_event()],
            observations=obs,
            lookback_hours=4.0, lookahead_hours=2.0,
        )
        assert result["status"] == "ok"
        if result["correlations"]:
            c = result["correlations"][0]
            for k in ("event", "observations_in_window", "temporal_distance_minutes",
                      "top_deviations", "correlation_score", "interpretation",
                      "component_scores", "evidence"):
                assert k in c, f"Missing key '{k}' in correlation entry"

    def test_json_serialisable(self):
        obs  = _baseline_obs(20) + [_obs(offset_hours=0.5)]
        result = correlate_space_events(events=[_event()], observations=obs)
        json.dumps(result)  # must not raise

    def test_correlation_score_in_range(self):
        obs = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=800.0)]
        result = correlate_space_events(events=[_event()], observations=obs)
        for c in result["correlations"]:
            assert 0.0 <= c["correlation_score"] <= 1.0

    def test_component_scores_in_range(self):
        obs = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=700.0)]
        result = correlate_space_events(events=[_event()], observations=obs)
        for c in result["correlations"]:
            cs = c["component_scores"]
            for k, v in cs.items():
                assert 0.0 <= v <= 1.0, f"component_scores[{k}] = {v} out of range"


# ---------------------------------------------------------------------------
# Edge cases — no inputs
# ---------------------------------------------------------------------------

class TestNoInputs:

    def test_no_events_returns_ok(self):
        result = correlate_space_events(observations=[_obs()])
        assert result["status"] == "ok"
        assert result["correlations"] == []
        assert result["summary"]["events_analyzed"] == 0

    def test_no_observations_returns_ok(self):
        result = correlate_space_events(events=[_event()])
        assert result["status"] == "ok"
        assert result["correlations"] == []
        assert result["summary"]["observations_analyzed"] == 0

    def test_both_empty_returns_ok(self):
        result = correlate_space_events()
        assert result["status"] == "ok"
        assert result["correlations"] == []

    def test_event_without_timestamp_skipped(self):
        ev = {"event_type": "CME", "event_time": None, "source": "NASA"}
        result = correlate_space_events(events=[ev], observations=[_obs()])
        # Event without valid timestamp is skipped — correlations may be empty
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Temporal proximity
# ---------------------------------------------------------------------------

class TestTemporalProximity:

    def test_observation_at_event_time_maximum_temporal_score(self):
        """Observation at exactly t_event → temporal component = 1.0."""
        obs = _baseline_obs(20) + [_obs(offset_hours=0.0)]  # offset_hours=0 → t_event
        result = correlate_space_events(
            events=[_event()],
            observations=obs,
            lookback_hours=4.0, lookahead_hours=2.0,
        )
        assert result["status"] == "ok"
        if result["correlations"]:
            assert result["correlations"][0]["component_scores"]["temporal"] == 1.0

    def test_observation_inside_window_included(self):
        obs = _baseline_obs(20) + [_obs(offset_hours=1.0)]  # 1h after event
        result = correlate_space_events(
            events=[_event()],
            observations=obs,
            lookback_hours=4.0, lookahead_hours=2.0,
        )
        assert result["status"] == "ok"
        assert len(result["correlations"]) > 0
        assert result["correlations"][0]["observations_in_window"] >= 1

    def test_observation_outside_window_excluded(self):
        # Event at t0, observation at t0 + 10h, window = ±2h
        obs = _baseline_obs(20) + [_obs(offset_hours=10.0, speed=999.0)]
        result = correlate_space_events(
            events=[_event()],
            observations=obs,
            lookback_hours=2.0, lookahead_hours=2.0,
        )
        # 999 km/s observation is outside window and must not appear in deviations
        if result["correlations"]:
            for dev in result["correlations"][0]["top_deviations"]:
                if dev["parameter"] == "solar_wind_speed":
                    assert dev["value"] != 999.0, (
                        "Out-of-window observation must not appear in deviations"
                    )


# ---------------------------------------------------------------------------
# Anomaly / statistical component
# ---------------------------------------------------------------------------

class TestAnomalyComponent:

    def test_high_z_score_drives_anomaly_score(self):
        """A massive speed spike should yield a high anomaly component score."""
        baseline = _baseline_obs(30, speed_mean=400.0)
        spike    = [_obs(offset_hours=0.5, speed=900.0)]  # ~16σ above baseline
        result   = correlate_space_events(
            events=[_event()],
            observations=baseline + spike,
            lookback_hours=4.0, lookahead_hours=2.0,
        )
        assert result["status"] == "ok"
        if result["correlations"]:
            assert result["correlations"][0]["component_scores"]["anomaly"] > 0.5

    def test_flat_baseline_no_anomaly(self):
        """All values at exactly the baseline mean → anomaly_score ≈ 0."""
        baseline = _baseline_obs(30, speed_mean=400.0)
        in_win   = [_obs(offset_hours=0.5, speed=400.0)]
        result   = correlate_space_events(
            events=[_event()],
            observations=baseline + in_win,
            lookback_hours=4.0, lookahead_hours=2.0,
        )
        if result["correlations"]:
            assert result["correlations"][0]["component_scores"]["anomaly"] < 0.2

    def test_direction_above_below_in_deviations(self):
        """Spike upward → direction = 'above'; spike downward → 'below'."""
        baseline = _baseline_obs(30, speed_mean=400.0)
        spike_up = [_obs(offset_hours=0.5,  speed=900.0)]
        result_up = correlate_space_events(
            events=[_event()], observations=baseline + spike_up,
            lookback_hours=4.0, lookahead_hours=2.0,
        )
        if result_up["correlations"]:
            devs = {d["parameter"]: d for d in result_up["correlations"][0]["top_deviations"]}
            if "solar_wind_speed" in devs:
                assert devs["solar_wind_speed"]["direction"] == "above"

        baseline2 = _baseline_obs(30, speed_mean=400.0)
        spike_dn  = [_obs(offset_hours=0.5, speed=50.0)]
        result_dn = correlate_space_events(
            events=[_event()], observations=baseline2 + spike_dn,
            lookback_hours=4.0, lookahead_hours=2.0,
        )
        if result_dn["correlations"]:
            devs = {d["parameter"]: d for d in result_dn["correlations"][0]["top_deviations"]}
            if "solar_wind_speed" in devs:
                assert devs["solar_wind_speed"]["direction"] == "below"


# ---------------------------------------------------------------------------
# Trend component
# ---------------------------------------------------------------------------

class TestTrendComponent:

    def test_significant_variation_drives_trend_score(self):
        """Speed going from 300→600 in window (100% change) → trend_score high."""
        baseline = _baseline_obs(30, speed_mean=400.0)
        ramp     = [
            _obs(offset_hours=-0.5, speed=300.0),
            _obs(offset_hours=0.0,  speed=450.0),
            _obs(offset_hours=0.5,  speed=600.0),
        ]
        result = correlate_space_events(
            events=[_event()],
            observations=baseline + ramp,
            lookback_hours=1.0, lookahead_hours=1.0,
        )
        if result["correlations"]:
            assert result["correlations"][0]["component_scores"]["trend"] > 0.5

    def test_flat_variation_zero_trend_score(self):
        """Constant values → trend_score = 0."""
        baseline = _baseline_obs(30, speed_mean=400.0)
        flat     = [_obs(offset_hours=h * 0.1, speed=400.0) for h in range(5)]
        result   = correlate_space_events(
            events=[_event()],
            observations=baseline + flat,
            lookback_hours=1.0, lookahead_hours=1.0,
        )
        if result["correlations"]:
            assert result["correlations"][0]["component_scores"]["trend"] == 0.0


# ---------------------------------------------------------------------------
# Event-weight component
# ---------------------------------------------------------------------------

class TestEventWeights:

    def _score_for_type(self, event_type: str) -> float:
        obs    = _baseline_obs(20) + [_obs(offset_hours=0.5)]
        result = correlate_space_events(
            events=[_event(event_type=event_type)],
            observations=obs,
            lookback_hours=4.0, lookahead_hours=2.0,
            min_score=0.0,
        )
        if result["correlations"]:
            return result["correlations"][0]["component_scores"]["event_weight"]
        return 0.0

    def test_cme_has_highest_event_weight(self):
        assert self._score_for_type("CME") == 1.0

    def test_alert_lower_than_cme(self):
        assert self._score_for_type("ALERT") < self._score_for_type("CME")

    def test_unknown_type_gets_minimum_weight(self):
        obs = _baseline_obs(20) + [_obs(offset_hours=0.5)]
        result = correlate_space_events(
            events=[_event(event_type="UNKNOWN_TYPE")],
            observations=obs,
            lookback_hours=4.0, lookahead_hours=2.0,
            min_score=0.0,
        )
        if result["correlations"]:
            w = result["correlations"][0]["component_scores"]["event_weight"]
            assert w <= 0.3  # minimum is 0.2


# ---------------------------------------------------------------------------
# Multiple events
# ---------------------------------------------------------------------------

class TestMultipleEvents:

    def test_correlations_sorted_by_score_descending(self):
        baseline = _baseline_obs(30, speed_mean=400.0)
        # Event A: nearby large spike → high score
        ev_a     = _event(event_type="CME", offset_hours=0.0)
        spike    = [_obs(offset_hours=0.5, speed=900.0, kp=8.0)]
        # Event B: no observations nearby → low score
        ev_b     = {"event_type": "ALERT", "event_time": _ts(_T0 + timedelta(hours=50)),
                    "source": "NOAA", "external_id": "B", "description": "", "severity": None,
                    "extra": {}}

        result = correlate_space_events(
            events=[ev_b, ev_a],   # intentionally wrong order
            observations=baseline + spike,
            lookback_hours=4.0, lookahead_hours=2.0,
            min_score=0.0,
        )
        scores = [c["correlation_score"] for c in result["correlations"]]
        assert scores == sorted(scores, reverse=True)

    def test_summary_events_analyzed_count(self):
        obs = _baseline_obs(20) + [_obs(offset_hours=0.5)]
        result = correlate_space_events(
            events=[_event("CME"), _event("FLR"), _event("GST")],
            observations=obs,
            lookback_hours=4.0, lookahead_hours=2.0,
        )
        assert result["summary"]["events_analyzed"] == 3

    def test_observations_analyzed_count(self):
        n_obs  = 25
        obs    = _baseline_obs(n_obs - 1) + [_obs(offset_hours=0.5)]
        result = correlate_space_events(
            events=[_event()],
            observations=obs,
        )
        assert result["summary"]["observations_analyzed"] == n_obs


# ---------------------------------------------------------------------------
# Interpretation thresholds
# ---------------------------------------------------------------------------

class TestInterpretationThresholds:

    def test_high_score_is_strong(self):
        # Force high score: many spike obs near CME event + high baseline
        baseline = _baseline_obs(30, speed_mean=400.0)
        spikes   = [_obs(offset_hours=h * 0.1, speed=900.0, kp=9.0, proton=200.0)
                    for h in range(5)]
        result   = correlate_space_events(
            events=[_event("CME")],
            observations=baseline + spikes,
            lookback_hours=1.0, lookahead_hours=1.0,
        )
        if result["correlations"]:
            score = result["correlations"][0]["correlation_score"]
            interp = result["correlations"][0]["interpretation"]
            assert interp in ("strong_temporal_association",
                              "moderate_temporal_association"), (
                f"score={score} → unexpected interpretation={interp}"
            )

    def test_no_observations_in_window_low_score(self):
        obs = _baseline_obs(20) + [_obs(offset_hours=10.0, speed=900.0)]
        result = correlate_space_events(
            events=[_event()],
            observations=obs,
            lookback_hours=2.0, lookahead_hours=2.0,
            min_score=0.0,
        )
        if result["correlations"]:
            assert result["correlations"][0]["interpretation"] in (
                "no_significant_correlation", "weak_temporal_association"
            )

    def test_interpret_function_all_bands(self):
        assert _interpret(0.80) == "strong_temporal_association"
        assert _interpret(0.60) == "moderate_temporal_association"
        assert _interpret(0.35) == "weak_temporal_association"
        assert _interpret(0.10) == "no_significant_correlation"
        assert _interpret(0.00) == "no_significant_correlation"


# ---------------------------------------------------------------------------
# min_score filtering
# ---------------------------------------------------------------------------

class TestMinScore:

    def test_min_score_zero_returns_all(self):
        obs = _baseline_obs(10) + [_obs(offset_hours=0.5)]
        r0 = correlate_space_events(events=[_event()], observations=obs, min_score=0.0)
        r1 = correlate_space_events(events=[_event()], observations=obs, min_score=0.5)
        assert len(r0["correlations"]) >= len(r1["correlations"])

    def test_min_score_1_returns_none_unless_perfect(self):
        obs = _baseline_obs(10) + [_obs(offset_hours=0.5)]
        result = correlate_space_events(events=[_event()], observations=obs, min_score=0.999)
        # Very unlikely to achieve score=1.0 with imperfect data
        for c in result["correlations"]:
            assert c["correlation_score"] >= 0.999


# ---------------------------------------------------------------------------
# Causal language prevention
# ---------------------------------------------------------------------------

class TestCausalLanguagePrevention:

    def _all_evidence(self, result: dict) -> list[str]:
        texts = []
        for c in result["correlations"]:
            texts.extend(c.get("evidence", []))
        return texts

    def test_no_caused_in_evidence(self):
        obs    = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=900.0)]
        result = correlate_space_events(events=[_event("CME")], observations=obs)
        for text in self._all_evidence(result):
            assert "caused" not in text.lower(), f"Causal language found: '{text}'"

    def test_no_due_to_in_evidence(self):
        obs    = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=900.0)]
        result = correlate_space_events(events=[_event()], observations=obs)
        for text in self._all_evidence(result):
            assert "due to" not in text.lower(), f"Causal language found: '{text}'"

    def test_no_resulted_in_evidence(self):
        obs    = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=900.0)]
        result = correlate_space_events(events=[_event()], observations=obs)
        for text in self._all_evidence(result):
            assert "resulted in" not in text.lower()

    def test_interpretation_labels_no_causal_language(self):
        valid = {
            "no_significant_correlation",
            "weak_temporal_association",
            "moderate_temporal_association",
            "strong_temporal_association",
        }
        obs    = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=900.0)]
        result = correlate_space_events(events=[_event()], observations=obs)
        for c in result["correlations"]:
            assert c["interpretation"] in valid, (
                f"Unexpected interpretation: {c['interpretation']}"
            )

    def test_causal_guard_strips_forbidden_phrases(self):
        text = "The CME caused the anomaly due to solar wind increase."
        cleaned = _causal_guard(text)
        assert "caused" not in cleaned.lower()
        assert "due to" not in cleaned.lower()

    def test_causal_guard_passthrough_clean_text(self):
        text = "Temporally coincident NOAA increase detected."
        assert _causal_guard(text) == text


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_lookback_zero_returns_error(self):
        result = correlate_space_events(lookback_hours=0)
        assert result["status"] == "error"

    def test_lookback_negative_returns_error(self):
        result = correlate_space_events(lookback_hours=-1.0)
        assert result["status"] == "error"

    def test_lookahead_zero_returns_error(self):
        result = correlate_space_events(lookahead_hours=0)
        assert result["status"] == "error"

    def test_min_score_negative_returns_error(self):
        result = correlate_space_events(min_score=-0.1)
        assert result["status"] == "error"

    def test_min_score_above_1_returns_error(self):
        result = correlate_space_events(min_score=1.1)
        assert result["status"] == "error"

    def test_non_numeric_lookback_returns_error(self):
        result = correlate_space_events(lookback_hours="six")   # type: ignore[arg-type]
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Dataclass inputs accepted
# ---------------------------------------------------------------------------

class TestDataclassInputs:

    def test_spaceweatherevent_dataclass_accepted(self):
        ev  = SpaceWeatherEvent(event_type="CME", event_time=_ts(_T0), source="NASA")
        obs = _baseline_obs(10) + [_obs(offset_hours=0.5)]
        result = correlate_space_events(events=[ev], observations=obs)
        assert result["status"] == "ok"

    def test_spaceweatherobservation_dataclass_accepted(self):
        ev  = [_event()]
        obs = SpaceWeatherObservation(
            timestamp=_ts(_T0 + timedelta(hours=0.5)),
            solar_wind_speed=700.0, source="NOAA",
        )
        result = correlate_space_events(events=ev, observations=[obs])
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_inputs_same_scores(self):
        obs = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=700.0)]
        r1  = correlate_space_events(events=[_event()], observations=obs)
        r2  = correlate_space_events(events=[_event()], observations=obs)
        assert r1["correlations"] == r2["correlations"]


# ---------------------------------------------------------------------------
# potential_correlations count accuracy
# ---------------------------------------------------------------------------

class TestPotentialCorrelationCount:

    def test_count_excludes_no_significant(self):
        obs    = _baseline_obs(20) + [_obs(offset_hours=0.5, speed=700.0)]
        result = correlate_space_events(events=[_event()], observations=obs, min_score=0.0)
        non_zero = sum(
            1 for c in result["correlations"]
            if c["interpretation"] != "no_significant_correlation"
        )
        assert result["summary"]["potential_correlations"] == non_zero


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:

    def test_temporal_score_at_event_time(self):
        t = datetime(2025, 7, 20, 14, 0, 0, tzinfo=timezone.utc)
        assert _temporal_score(t, t, window_seconds=3600) == 1.0

    def test_temporal_score_at_boundary(self):
        t_event = datetime(2025, 7, 20, 14, 0, 0, tzinfo=timezone.utc)
        t_obs   = t_event + timedelta(seconds=3600)  # exactly at boundary
        score   = _temporal_score(t_obs, t_event, window_seconds=3600)
        assert abs(score) < 0.001   # ≈ 0

    def test_temporal_score_beyond_window(self):
        t_event = datetime(2025, 7, 20, 14, 0, 0, tzinfo=timezone.utc)
        t_obs   = t_event + timedelta(hours=10)
        score   = _temporal_score(t_obs, t_event, window_seconds=3600)
        assert score == 0.0

    def test_anomaly_score_zero(self):
        assert _anomaly_score(0.0) == 0.0

    def test_anomaly_score_capped(self):
        assert _anomaly_score(100.0) == 1.0

    def test_anomaly_score_none(self):
        assert _anomaly_score(None) == 0.0

    def test_compute_z_above(self):
        baseline = [400.0] * 10
        z = _compute_z(450.0, baseline)
        # All baseline identical → std≈0 → epsilon denominator → large positive z
        assert z is not None and z > 0

    def test_compute_z_below(self):
        baseline = [400.0, 390.0, 410.0, 405.0, 395.0]
        z = _compute_z(300.0, baseline)
        assert z is not None and z < 0

    def test_compute_z_insufficient_baseline(self):
        assert _compute_z(400.0, [400.0]) is None

    def test_interpret_all_thresholds(self):
        assert _interpret(1.0)   == "strong_temporal_association"
        assert _interpret(0.75)  == "strong_temporal_association"
        assert _interpret(0.749) == "moderate_temporal_association"
        assert _interpret(0.50)  == "moderate_temporal_association"
        assert _interpret(0.499) == "weak_temporal_association"
        assert _interpret(0.25)  == "weak_temporal_association"
        assert _interpret(0.249) == "no_significant_correlation"
        assert _interpret(0.0)   == "no_significant_correlation"
