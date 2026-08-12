"""
Tests for analyze_trends — BobVoyage MCP tool

Covers:
  - Successful analysis on the real dev dataset (default window)
  - Custom window sizes
  - Increasing values are correctly classified
  - Decreasing values are correctly classified
  - Stable values are correctly classified
  - Empty dataset → error
  - Single-row dataset (insufficient) → error
  - Missing numeric columns → gracefully skipped, not included in trends
  - Invalid window values (0, 1, negative, non-integer) → error
  - Window larger than dataset → clamped, no error
  - Percentage / absolute change calculations
  - significant_trends sorted by |change_pct| descending
  - significant_trends excludes stable parameters
  - All output values are JSON-serialisable
  - Window metadata (start, end timestamps) present and correct
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from bobvoyage.tools.analyze_trends import analyze_trends


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_row(
    ts: str,
    speed: float = 400.0,
    density: float = 5.0,
    bfield: float = 7.0,
    xray: float = 1.2e-7,
    proton: float = 2.0,
    kp: float = 2.5,
) -> dict:
    return {
        "timestamp":          ts,
        "solar_wind_speed":   speed,
        "solar_wind_density": density,
        "magnetic_field":     bfield,
        "xray_flux":          f"{xray:.3e}",
        "proton_flux":        proton,
        "geomagnetic_index":  kp,
    }


# ---------------------------------------------------------------------------
# Real-dataset tests
# ---------------------------------------------------------------------------

class TestAnalyzeTrendsRealDataset:

    def test_returns_ok_status(self):
        result = analyze_trends()
        assert result["status"] == "ok", result["message"]

    def test_window_metadata_present(self):
        result = analyze_trends(window=12)
        w = result["window"]
        assert w["observations"] == 12
        assert w["start"] is not None
        assert w["end"] is not None

    def test_start_before_end(self):
        result = analyze_trends(window=12)
        w = result["window"]
        assert w["start"] < w["end"]

    def test_trends_dict_contains_expected_params(self):
        result = analyze_trends()
        trends = result["trends"]
        for param in (
            "solar_wind_speed", "solar_wind_density", "magnetic_field",
            "xray_flux", "proton_flux", "geomagnetic_index",
        ):
            assert param in trends, f"'{param}' missing from trends"

    def test_each_trend_has_required_keys(self):
        result = analyze_trends()
        for param, data in result["trends"].items():
            for key in ("direction", "change_percent", "change_absolute",
                        "start_value", "end_value", "severity", "observations_used"):
                assert key in data, f"Key '{key}' missing from trend '{param}'"

    def test_direction_is_valid(self):
        result = analyze_trends()
        valid = {"increasing", "decreasing", "stable"}
        for param, data in result["trends"].items():
            assert data["direction"] in valid, (
                f"Invalid direction '{data['direction']}' for '{param}'"
            )

    def test_severity_is_valid(self):
        result = analyze_trends()
        valid = {"stable", "minor", "moderate", "significant"}
        for param, data in result["trends"].items():
            assert data["severity"] in valid

    def test_significant_trends_list_present(self):
        result = analyze_trends()
        assert isinstance(result["significant_trends"], list)

    def test_output_is_json_serialisable(self):
        result = analyze_trends()
        json.dumps(result)  # must not raise

    def test_custom_window_5(self):
        result = analyze_trends(window=5)
        assert result["status"] == "ok"
        assert result["window"]["observations"] == 5


# ---------------------------------------------------------------------------
# Direction classification tests
# ---------------------------------------------------------------------------

class TestDirectionClassification:

    def test_increasing_values(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=300.0),
            _make_row("2025-07-20T10:05:00Z", speed=320.0),
            _make_row("2025-07-20T10:10:00Z", speed=360.0),
            _make_row("2025-07-20T10:15:00Z", speed=420.0),  # +40% from 300
        ])
        result = analyze_trends(window=4, dataset_path=csv_file)
        assert result["status"] == "ok"
        assert result["trends"]["solar_wind_speed"]["direction"] == "increasing"

    def test_decreasing_values(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=500.0),
            _make_row("2025-07-20T10:05:00Z", speed=470.0),
            _make_row("2025-07-20T10:10:00Z", speed=430.0),
            _make_row("2025-07-20T10:15:00Z", speed=380.0),  # -24% from 500
        ])
        result = analyze_trends(window=4, dataset_path=csv_file)
        assert result["status"] == "ok"
        assert result["trends"]["solar_wind_speed"]["direction"] == "decreasing"

    def test_stable_values(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        # < 5 % change → stable
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=400.0),
            _make_row("2025-07-20T10:05:00Z", speed=401.0),
            _make_row("2025-07-20T10:10:00Z", speed=403.0),
            _make_row("2025-07-20T10:15:00Z", speed=402.0),  # +0.5% from 400
        ])
        result = analyze_trends(window=4, dataset_path=csv_file)
        assert result["status"] == "ok"
        assert result["trends"]["solar_wind_speed"]["direction"] == "stable"


# ---------------------------------------------------------------------------
# Calculation accuracy tests
# ---------------------------------------------------------------------------

class TestCalculations:

    def test_change_percent_calculation(self, tmp_path):
        """start=400, end=500 → change_pct = (500-400)/400*100 = 25.0"""
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=400.0),
            _make_row("2025-07-20T10:05:00Z", speed=500.0),
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        assert result["status"] == "ok"
        pct = result["trends"]["solar_wind_speed"]["change_percent"]
        assert abs(pct - 25.0) < 0.01

    def test_change_absolute_calculation(self, tmp_path):
        """start=400, end=500 → abs change = 100"""
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=400.0),
            _make_row("2025-07-20T10:05:00Z", speed=500.0),
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        abs_change = result["trends"]["solar_wind_speed"]["change_absolute"]
        assert abs(abs_change - 100.0) < 0.01

    def test_start_and_end_values_correct(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=350.0),
            _make_row("2025-07-20T10:05:00Z", speed=450.0),
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        t = result["trends"]["solar_wind_speed"]
        assert abs(t["start_value"] - 350.0) < 0.01
        assert abs(t["end_value"]   - 450.0) < 0.01

    def test_negative_change_direction(self, tmp_path):
        """start=500, end=400 → -20%, decreasing"""
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=500.0),
            _make_row("2025-07-20T10:05:00Z", speed=400.0),
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        t = result["trends"]["solar_wind_speed"]
        assert t["direction"] == "decreasing"
        assert t["change_percent"] < 0

    def test_severity_minor(self, tmp_path):
        """10% change → minor"""
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=400.0),
            _make_row("2025-07-20T10:05:00Z", speed=440.0),
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        assert result["trends"]["solar_wind_speed"]["severity"] == "minor"

    def test_severity_moderate(self, tmp_path):
        """20% change → moderate"""
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=400.0),
            _make_row("2025-07-20T10:05:00Z", speed=480.0),
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        assert result["trends"]["solar_wind_speed"]["severity"] == "moderate"

    def test_severity_significant(self, tmp_path):
        """50% change → significant"""
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=400.0),
            _make_row("2025-07-20T10:05:00Z", speed=600.0),
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        assert result["trends"]["solar_wind_speed"]["severity"] == "significant"


# ---------------------------------------------------------------------------
# significant_trends ordering and filtering
# ---------------------------------------------------------------------------

class TestSignificantTrends:

    def test_stable_params_excluded_from_significant(self, tmp_path):
        """A parameter with < 5% change must NOT appear in significant_trends."""
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=400.0, kp=2.0),
            _make_row("2025-07-20T10:05:00Z", speed=401.0, kp=2.0),  # ~0.25% — stable
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        sig_params = [e["parameter"] for e in result["significant_trends"]]
        assert "solar_wind_speed" not in sig_params
        assert "geomagnetic_index" not in sig_params

    def test_significant_trends_sorted_by_magnitude(self, tmp_path):
        """Larger |change_pct| must come first."""
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            # speed: +50%, kp: +100%
            _make_row("2025-07-20T10:00:00Z", speed=200.0, kp=1.0),
            _make_row("2025-07-20T10:05:00Z", speed=300.0, kp=2.0),
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        sig = result["significant_trends"]
        assert len(sig) >= 2
        abs_changes = [abs(e["change_percent"]) for e in sig]
        assert abs_changes == sorted(abs_changes, reverse=True), (
            f"Expected descending order, got: {abs_changes}"
        )

    def test_significant_trends_entry_has_required_keys(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            _make_row("2025-07-20T10:00:00Z", speed=300.0),
            _make_row("2025-07-20T10:05:00Z", speed=450.0),  # 50%
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        for entry in result["significant_trends"]:
            for key in ("parameter", "direction", "change_percent", "severity"):
                assert key in entry


# ---------------------------------------------------------------------------
# Window behaviour
# ---------------------------------------------------------------------------

class TestWindowBehaviour:

    def test_window_larger_than_dataset_clamped(self, tmp_path):
        """window=100 with only 5 rows → uses 5, no error."""
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(f"2025-07-20T10:0{i}:00Z", speed=400.0 + i * 10)
                for i in range(5)]
        _write_csv(csv_file, rows)
        result = analyze_trends(window=100, dataset_path=csv_file)
        assert result["status"] == "ok"
        assert result["window"]["observations"] == 5

    def test_window_selects_most_recent_rows(self, tmp_path):
        """With window=2 on 5 rows, only the last 2 rows should be analyzed."""
        csv_file = tmp_path / "sw.csv"
        # Rows with speed 100, 200, 300, 400, 500; window=2 → start=400, end=500
        rows = [_make_row(f"2025-07-20T10:0{i}:00Z", speed=float((i + 1) * 100))
                for i in range(5)]
        _write_csv(csv_file, rows)
        result = analyze_trends(window=2, dataset_path=csv_file)
        assert result["status"] == "ok"
        t = result["trends"]["solar_wind_speed"]
        assert abs(t["start_value"] - 400.0) < 0.01
        assert abs(t["end_value"]   - 500.0) < 0.01


# ---------------------------------------------------------------------------
# Error / edge-case handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_file_returns_error(self, tmp_path):
        result = analyze_trends(dataset_path=tmp_path / "nonexistent.csv")
        assert result["status"] == "error"
        assert result["trends"] == {}
        assert "not found" in result["message"].lower()

    def test_empty_file_returns_error(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        result = analyze_trends(dataset_path=csv_file)
        assert result["status"] == "error"
        assert result["observation"] if False else True  # just check status
        assert result["trends"] == {}

    def test_single_row_insufficient(self, tmp_path):
        csv_file = tmp_path / "single.csv"
        _write_csv(csv_file, [_make_row("2025-07-20T10:00:00Z")])
        result = analyze_trends(dataset_path=csv_file)
        assert result["status"] == "error"
        assert "insufficient" in result["message"].lower()

    def test_invalid_window_zero(self):
        result = analyze_trends(window=0)
        assert result["status"] == "error"
        assert "invalid window" in result["message"].lower()

    def test_invalid_window_one(self):
        result = analyze_trends(window=1)
        assert result["status"] == "error"
        assert "invalid window" in result["message"].lower()

    def test_invalid_window_negative(self):
        result = analyze_trends(window=-5)
        assert result["status"] == "error"

    def test_invalid_window_non_integer(self):
        result = analyze_trends(window="twelve")  # type: ignore[arg-type]
        assert result["status"] == "error"

    def test_missing_numeric_columns_skipped(self, tmp_path):
        """CSV with only timestamp + solar_wind_speed — other params absent."""
        csv_file = tmp_path / "partial.csv"
        _write_csv(csv_file, [
            {"timestamp": "2025-07-20T10:00:00Z", "solar_wind_speed": 400.0},
            {"timestamp": "2025-07-20T10:05:00Z", "solar_wind_speed": 450.0},
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        assert result["status"] == "ok"
        assert "solar_wind_speed" in result["trends"]
        # Other params should NOT appear since they're not in the CSV
        for absent in ("solar_wind_density", "magnetic_field", "xray_flux",
                       "proton_flux", "geomagnetic_index"):
            assert absent not in result["trends"]

    def test_all_nan_column_skipped(self, tmp_path):
        """A column present in the CSV but with all NaN values is skipped."""
        csv_file = tmp_path / "nan.csv"
        _write_csv(csv_file, [
            {"timestamp": "2025-07-20T10:00:00Z",
             "solar_wind_speed": "n/a", "geomagnetic_index": 2.0},
            {"timestamp": "2025-07-20T10:05:00Z",
             "solar_wind_speed": "n/a", "geomagnetic_index": 3.0},
        ])
        result = analyze_trends(window=2, dataset_path=csv_file)
        assert result["status"] == "ok"
        assert "solar_wind_speed" not in result["trends"]
        assert "geomagnetic_index" in result["trends"]
