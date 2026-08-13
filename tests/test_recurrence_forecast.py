"""
Tests for recurrence_forecast — BobVoyage MCP tool

Covers:
  - Output schema validation (all required top-level and per-region keys)
  - Active region 13664 (historical X8.7 producer, May 2024) → status="ok",
    risk_level="elevated_recurrence_risk"
  - Active region not present in the dataset → status="error"
  - Parameter validation: lookback_days <= 0 → error
  - Parameter validation: min_flares < 1 → error
  - Bad dataset_path → error
  - Full scan (no active_region) returns a list sorted by risk_score descending
  - Full scan only includes regions with >= min_flares flares
  - risk_score is always within [0, 100]
  - risk_level values are restricted to the three documented labels
  - JSON-serialisable output for both specific-region and full-scan calls
  - Determinism: same inputs → same outputs on repeated calls
  - evidence list is non-empty for regions with valid source_location data
  - forecast_reentry_window contains "start" and "end" ISO strings when
    position data is available
  - as_of parameter correctly anchors the lookback window
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bobvoyage.tools.recurrence_forecast import (
    recurrence_forecast,
    _parse_source_location,
    _flare_score,
    _is_strong_flare,
    _classify_risk,
)

# ---------------------------------------------------------------------------
# Dataset path (real CSV, path relative to BobVoyage project root)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATASET = _PROJECT_ROOT / "data" / "space_weather_unified.csv"

# Region 13664: historically produced X8.7 on 2024-05-14; well-sampled in CSV
_KNOWN_REGION = 13664

# An active_region integer that does not exist in the dataset
_UNKNOWN_REGION = 99999

# A fixed as_of timestamp that places region 13664 firmly inside a generous
# lookback window, avoiding any dependence on "today"
_AS_OF_REGION_13664 = "2024-05-15T12:00:00+00:00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_region(result: dict, ar: int) -> dict | None:
    """Return the region entry for `ar` from result["regions"], or None."""
    for r in result["regions"]:
        if r["active_region"] == ar:
            return r
    return None


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def test_top_level_keys_present_specific_region(self):
        result = recurrence_forecast(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=90,
            as_of=_AS_OF_REGION_13664,
        )
        for key in ("status", "as_of", "parameters", "regions", "message"):
            assert key in result, f"Top-level key '{key}' missing"

    def test_parameters_echoed_in_output(self):
        result = recurrence_forecast(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=90,
            min_flares=2,
            as_of=_AS_OF_REGION_13664,
        )
        assert result["parameters"]["active_region"] == _KNOWN_REGION
        assert result["parameters"]["lookback_days"] == 90
        assert result["parameters"]["min_flares"] == 2

    def test_region_entry_keys_present(self):
        result = recurrence_forecast(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=90,
            as_of=_AS_OF_REGION_13664,
        )
        assert result["status"] == "ok"
        assert len(result["regions"]) == 1
        region = result["regions"][0]
        expected_keys = (
            "active_region", "last_seen", "last_position", "flare_count",
            "strongest_flare", "productivity_score", "days_to_limb_exit",
            "forecast_reentry_window", "risk_score", "risk_level", "evidence",
        )
        for key in expected_keys:
            assert key in region, f"Region key '{key}' missing"

    def test_as_of_is_iso_string(self):
        result = recurrence_forecast(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=90,
            as_of=_AS_OF_REGION_13664,
        )
        assert isinstance(result["as_of"], str)
        # Must be parseable as a timestamp
        import pandas as pd
        pd.to_datetime(result["as_of"])


# ---------------------------------------------------------------------------
# Known historical case: active region 13664
# ---------------------------------------------------------------------------

class TestKnownRegion13664:
    """
    Region 13664 produced an X8.7 flare on 2024-05-14 and dozens of M/X class
    flares across May 2–15, 2024. With a 90-day lookback anchored at
    2024-05-15T12:00Z it must be analysed and classified as elevated.
    """

    def _result(self) -> dict:
        return recurrence_forecast(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=90,
            as_of=_AS_OF_REGION_13664,
        )

    def test_status_ok(self):
        assert self._result()["status"] == "ok"

    def test_region_present_in_output(self):
        result = self._result()
        assert len(result["regions"]) == 1
        assert result["regions"][0]["active_region"] == _KNOWN_REGION

    def test_risk_level_elevated(self):
        region = self._result()["regions"][0]
        assert region["risk_level"] == "elevated_recurrence_risk", (
            f"Expected elevated_recurrence_risk, got {region['risk_level']} "
            f"(risk_score={region['risk_score']})"
        )

    def test_risk_score_above_threshold(self):
        region = self._result()["regions"][0]
        # elevated_recurrence_risk threshold is >= 60
        assert region["risk_score"] >= 60.0

    def test_flare_count_matches_csv(self):
        # CSV contains 95 lines matching 13664, not all Solar Flares with
        # valid active_region; exact count depends on parsing, but must be > 20
        region = self._result()["regions"][0]
        assert region["flare_count"] > 20

    def test_strongest_flare_is_x_class(self):
        region = self._result()["regions"][0]
        assert region["strongest_flare"] is not None
        assert region["strongest_flare"][0].upper() == "X"

    def test_forecast_reentry_window_present(self):
        region = self._result()["regions"][0]
        rw = region["forecast_reentry_window"]
        assert rw is not None, "forecast_reentry_window should be set for 13664"
        assert "start" in rw
        assert "end" in rw

    def test_forecast_reentry_window_start_before_end(self):
        import pandas as pd
        region = self._result()["regions"][0]
        rw = region["forecast_reentry_window"]
        start = pd.to_datetime(rw["start"])
        end = pd.to_datetime(rw["end"])
        assert start < end

    def test_evidence_non_empty(self):
        region = self._result()["regions"][0]
        assert len(region["evidence"]) > 0

    def test_evidence_contains_observed_tag(self):
        region = self._result()["regions"][0]
        assert any("OBSERVED" in e for e in region["evidence"])


# ---------------------------------------------------------------------------
# Unknown / absent active region
# ---------------------------------------------------------------------------

class TestUnknownRegion:

    def test_unknown_region_returns_error(self):
        result = recurrence_forecast(
            active_region=_UNKNOWN_REGION,
            dataset_path=_DATASET,
        )
        assert result["status"] == "error"

    def test_error_message_non_empty(self):
        result = recurrence_forecast(
            active_region=_UNKNOWN_REGION,
            dataset_path=_DATASET,
        )
        assert isinstance(result["message"], str)
        assert len(result["message"]) > 0

    def test_error_response_has_empty_regions(self):
        result = recurrence_forecast(
            active_region=_UNKNOWN_REGION,
            dataset_path=_DATASET,
        )
        assert result["regions"] == []


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestParameterValidation:

    def test_lookback_days_zero_returns_error(self):
        result = recurrence_forecast(
            dataset_path=_DATASET,
            lookback_days=0,
        )
        assert result["status"] == "error"

    def test_lookback_days_negative_returns_error(self):
        result = recurrence_forecast(
            dataset_path=_DATASET,
            lookback_days=-5.0,
        )
        assert result["status"] == "error"

    def test_min_flares_zero_returns_error(self):
        result = recurrence_forecast(
            dataset_path=_DATASET,
            min_flares=0,
        )
        assert result["status"] == "error"

    def test_min_flares_negative_returns_error(self):
        result = recurrence_forecast(
            dataset_path=_DATASET,
            min_flares=-1,
        )
        assert result["status"] == "error"

    def test_bad_dataset_path_returns_error(self):
        result = recurrence_forecast(
            dataset_path="/nonexistent/path/no_such_file.csv",
        )
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Full scan (no active_region specified)
# ---------------------------------------------------------------------------

class TestFullScan:

    def _scan(self, lookback_days: float = 45.0, min_flares: int = 2) -> dict:
        return recurrence_forecast(
            dataset_path=_DATASET,
            lookback_days=lookback_days,
            min_flares=min_flares,
            as_of=_AS_OF_REGION_13664,
        )

    def test_full_scan_returns_ok(self):
        result = self._scan()
        assert result["status"] == "ok"

    def test_full_scan_returns_list(self):
        result = self._scan()
        assert isinstance(result["regions"], list)

    def test_full_scan_sorted_by_risk_score_descending(self):
        result = self._scan()
        scores = [r["risk_score"] for r in result["regions"]]
        assert scores == sorted(scores, reverse=True), (
            f"Regions not sorted by risk_score desc: {scores}"
        )

    def test_full_scan_respects_min_flares(self):
        """All returned regions must have flare_count >= min_flares."""
        min_f = 3
        result = recurrence_forecast(
            dataset_path=_DATASET,
            lookback_days=45,
            min_flares=min_f,
            as_of=_AS_OF_REGION_13664,
        )
        assert result["status"] == "ok"
        for r in result["regions"]:
            assert r["flare_count"] >= min_f, (
                f"Region {r['active_region']} has flare_count={r['flare_count']} "
                f"but min_flares={min_f}"
            )

    def test_full_scan_active_region_param_is_none(self):
        result = self._scan()
        assert result["parameters"]["active_region"] is None

    def test_full_scan_risk_scores_in_range(self):
        result = self._scan()
        for r in result["regions"]:
            assert 0.0 <= r["risk_score"] <= 100.0

    def test_full_scan_risk_levels_valid(self):
        valid = {"low_recurrence_risk", "moderate_recurrence_risk", "elevated_recurrence_risk"}
        result = self._scan()
        for r in result["regions"]:
            assert r["risk_level"] in valid, (
                f"Unexpected risk_level '{r['risk_level']}' for region {r['active_region']}"
            )


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------

class TestJsonSerialisable:

    def test_specific_region_json_serialisable(self):
        result = recurrence_forecast(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=90,
            as_of=_AS_OF_REGION_13664,
        )
        json.dumps(result)  # must not raise

    def test_full_scan_json_serialisable(self):
        result = recurrence_forecast(
            dataset_path=_DATASET,
            lookback_days=45,
            as_of=_AS_OF_REGION_13664,
        )
        json.dumps(result)  # must not raise

    def test_error_response_json_serialisable(self):
        result = recurrence_forecast(
            active_region=_UNKNOWN_REGION,
            dataset_path=_DATASET,
        )
        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_specific_region_same_inputs_same_outputs(self):
        kwargs = dict(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=90,
            as_of=_AS_OF_REGION_13664,
        )
        r1 = recurrence_forecast(**kwargs)
        r2 = recurrence_forecast(**kwargs)
        assert r1["regions"] == r2["regions"]
        assert r1["as_of"] == r2["as_of"]

    def test_full_scan_same_inputs_same_outputs(self):
        kwargs = dict(
            dataset_path=_DATASET,
            lookback_days=45,
            as_of=_AS_OF_REGION_13664,
        )
        r1 = recurrence_forecast(**kwargs)
        r2 = recurrence_forecast(**kwargs)
        assert r1["regions"] == r2["regions"]

    def test_as_of_anchors_results(self):
        """Shifting as_of by a full synodic rotation should change risk scores
        (recency penalty changes even if same flares are in window)."""
        r_early = recurrence_forecast(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=400,
            as_of="2024-05-15T12:00:00+00:00",
        )
        r_late = recurrence_forecast(
            active_region=_KNOWN_REGION,
            dataset_path=_DATASET,
            lookback_days=400,
            as_of="2024-12-31T12:00:00+00:00",
        )
        # Both should succeed; the later as_of should carry a higher recency
        # penalty, meaning the late risk_score <= the early risk_score
        assert r_early["status"] == "ok"
        assert r_late["status"] == "ok"
        score_early = r_early["regions"][0]["risk_score"]
        score_late = r_late["regions"][0]["risk_score"]
        assert score_early >= score_late, (
            f"Expected earlier as_of to yield higher score "
            f"({score_early} vs {score_late})"
        )


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:

    def test_parse_source_location_north_east(self):
        loc = _parse_source_location("N25E90")
        assert loc is not None
        assert loc["lat_deg"] == pytest.approx(25.0)
        assert loc["lon_position_deg"] == pytest.approx(-90.0)  # E → negative

    def test_parse_source_location_south_west(self):
        loc = _parse_source_location("S19W59")
        assert loc is not None
        assert loc["lat_deg"] == pytest.approx(-19.0)
        assert loc["lon_position_deg"] == pytest.approx(59.0)  # W → positive

    def test_parse_source_location_central_meridian(self):
        loc = _parse_source_location("N00W00")
        assert loc is not None
        assert loc["lon_position_deg"] == pytest.approx(0.0)

    def test_parse_source_location_invalid_returns_none(self):
        assert _parse_source_location("UNKNOWN") is None
        assert _parse_source_location("") is None
        assert _parse_source_location(None) is None  # type: ignore[arg-type]

    def test_flare_score_x_class(self):
        # X1.0 → 1e-4 * 1.0 = 1e-4
        assert _flare_score("X1.0") == pytest.approx(1e-4)

    def test_flare_score_m_class(self):
        # M5.0 → 1e-5 * 5.0 = 5e-5
        assert _flare_score("M5.0") == pytest.approx(5e-5)

    def test_flare_score_x_greater_than_m(self):
        assert _flare_score("X1.0") > _flare_score("M9.9")

    def test_flare_score_unknown_letter_returns_zero(self):
        assert _flare_score("Z3.0") == 0.0

    def test_flare_score_none_returns_zero(self):
        assert _flare_score(None) == 0.0  # type: ignore[arg-type]

    def test_is_strong_flare_x_class(self):
        assert _is_strong_flare("X1.0") is True
        assert _is_strong_flare("X8.7") is True

    def test_is_strong_flare_m5_plus(self):
        assert _is_strong_flare("M5.0") is True
        assert _is_strong_flare("M9.9") is True

    def test_is_strong_flare_below_m5(self):
        assert _is_strong_flare("M4.9") is False
        assert _is_strong_flare("C9.9") is False

    def test_classify_risk_thresholds(self):
        assert _classify_risk(0.0)   == "low_recurrence_risk"
        assert _classify_risk(29.9)  == "low_recurrence_risk"
        assert _classify_risk(30.0)  == "moderate_recurrence_risk"
        assert _classify_risk(59.9)  == "moderate_recurrence_risk"
        assert _classify_risk(60.0)  == "elevated_recurrence_risk"
        assert _classify_risk(100.0) == "elevated_recurrence_risk"
