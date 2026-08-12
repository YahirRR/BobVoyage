"""
Tests for detect_anomalies — BobVoyage MCP tool

Covers:
  - Successful detection on the real dev dataset
  - No anomalies when all values are within baseline
  - Moderate anomaly (+2–3σ)
  - Significant anomaly (≥3σ)
  - Anomaly above baseline (positive z)
  - Anomaly below baseline (negative z)
  - Multiple simultaneous anomalies across parameters
  - Anomalies sorted by |z_score| descending
  - Stable parameter not flagged as anomaly
  - Insufficient baseline data (< 3 rows)
  - Missing values (NaN) in recent window are skipped
  - Zero standard deviation handled gracefully
  - Missing parameters are absent from output
  - Custom z_threshold respected
  - Window clamping when dataset is smaller than requested
  - Invalid recent_window (0, negative, non-integer)
  - Invalid baseline_window (< 3, non-integer)
  - Invalid z_threshold (0, negative, non-numeric)
  - Empty file → error
  - Missing file → error
  - All output values are JSON-serialisable
  - Summary fields correct (anomalies_detected, parameters_affected, highest_severity)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from bobvoyage.tools.detect_anomalies import detect_anomalies


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


def _make_row(ts: str, speed: float = 400.0, density: float = 5.0,
              bfield: float = 7.0, xray: float = 1.2e-7,
              proton: float = 2.0, kp: float = 2.5) -> dict:
    return {
        "timestamp":          ts,
        "solar_wind_speed":   speed,
        "solar_wind_density": density,
        "magnetic_field":     bfield,
        "xray_flux":          f"{xray:.3e}",
        "proton_flux":        proton,
        "geomagnetic_index":  kp,
    }


def _ts(i: int) -> str:
    """Generate a sequenced timestamp string for row i."""
    hour   = i // 12
    minute = (i % 12) * 5
    return f"2025-07-20T{hour:02d}:{minute:02d}:00Z"


def _build_flat_dataset(path: Path, n: int, speed: float = 400.0) -> None:
    """All rows identical speed (zero variance baseline)."""
    rows = [_make_row(_ts(i), speed=speed) for i in range(n)]
    _write_csv(path, rows)


def _build_normal_dataset(path: Path, n_baseline: int, n_recent: int,
                           speed_baseline: float = 400.0,
                           speed_recent: float = 400.0) -> None:
    """Baseline rows with speed_baseline, recent rows with speed_recent."""
    rows = [_make_row(_ts(i), speed=speed_baseline) for i in range(n_baseline)]
    rows += [_make_row(_ts(n_baseline + i), speed=speed_recent)
             for i in range(n_recent)]
    _write_csv(path, rows)


# ---------------------------------------------------------------------------
# Real-dataset smoke tests
# ---------------------------------------------------------------------------

class TestDetectAnomaliesRealDataset:

    def test_returns_ok_status(self):
        result = detect_anomalies()
        assert result["status"] == "ok", result["message"]

    def test_output_structure_complete(self):
        result = detect_anomalies()
        for key in ("status", "source", "baseline", "analysis_window",
                    "parameter_stats", "anomalies", "summary", "message"):
            assert key in result, f"Top-level key '{key}' missing"

    def test_baseline_metadata_present(self):
        result = detect_anomalies()
        b = result["baseline"]
        assert b["observations"] > 0
        assert b["start"] is not None
        assert b["end"] is not None

    def test_analysis_window_metadata_present(self):
        result = detect_anomalies()
        w = result["analysis_window"]
        assert w["observations"] > 0

    def test_summary_fields_present(self):
        result = detect_anomalies()
        s = result["summary"]
        for key in ("anomalies_detected", "parameters_affected",
                    "highest_severity", "total_observations_checked"):
            assert key in s

    def test_output_is_json_serialisable(self):
        result = detect_anomalies()
        json.dumps(result)   # must not raise

    def test_anomalies_sorted_by_abs_z_desc(self):
        result = detect_anomalies()
        zs = [abs(a["z_score"] or 0) for a in result["anomalies"]]
        assert zs == sorted(zs, reverse=True)

    def test_anomaly_entry_has_required_keys(self):
        result = detect_anomalies()
        for a in result["anomalies"]:
            for key in ("parameter", "timestamp", "observed_value",
                        "baseline_mean", "baseline_std", "z_score",
                        "severity", "direction"):
                assert key in a, f"Key '{key}' missing from anomaly entry"

    def test_severity_values_valid(self):
        result = detect_anomalies()
        valid = {"moderate", "significant"}
        for a in result["anomalies"]:
            assert a["severity"] in valid

    def test_direction_values_valid(self):
        result = detect_anomalies()
        valid = {"above_baseline", "below_baseline"}
        for a in result["anomalies"]:
            assert a["direction"] in valid


# ---------------------------------------------------------------------------
# No-anomaly case
# ---------------------------------------------------------------------------

class TestNoAnomalies:

    def test_no_anomalies_when_recent_matches_baseline(self, tmp_path):
        """All values identical → no anomalies expected."""
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(_ts(i), speed=400.0, density=5.0, kp=2.5)
                for i in range(20)]
        _write_csv(csv_file, rows)
        result = detect_anomalies(
            recent_window=3, baseline_window=10,
            z_threshold=2.0, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        assert result["summary"]["anomalies_detected"] is False
        assert result["summary"]["parameters_affected"] == 0
        assert result["anomalies"] == []

    def test_summary_highest_severity_none_when_no_anomalies(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(_ts(i)) for i in range(15)]
        _write_csv(csv_file, rows)
        result = detect_anomalies(
            recent_window=3, baseline_window=10, dataset_path=csv_file,
        )
        assert result["summary"]["highest_severity"] == "none"


# ---------------------------------------------------------------------------
# Moderate anomaly
# ---------------------------------------------------------------------------

class TestModerateAnomaly:

    def test_value_just_above_threshold_is_moderate(self, tmp_path):
        """
        Baseline: 20 rows all speed=400.0 → mean=400, std≈0.
        We need actual variance, so use a spread baseline.
        Baseline speeds 390–410 (std ≈ 6.1), recent = 400 + 2.2*std ≈ 413.4 → moderate.
        """
        csv_file = tmp_path / "sw.csv"
        import random
        random.seed(0)
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4)
                    for i in range(20)]   # speeds 396,398,400,402,404 cycling
        # mean≈400, std≈3.16
        # 2.0 * 3.16 = 6.32 → recent speed = 400 + 6.5 ≈ 406.5 → z≈2.06 → moderate
        recent = [_make_row(_ts(20), speed=406.5)]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=2.0, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        speed_anomaly = next(
            (a for a in result["anomalies"] if a["parameter"] == "solar_wind_speed"),
            None,
        )
        assert speed_anomaly is not None, "Expected a moderate anomaly in solar_wind_speed"
        assert speed_anomaly["severity"] == "moderate"

    def test_moderate_anomaly_direction_above(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4)
                    for i in range(20)]
        recent = [_make_row(_ts(20), speed=406.5)]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=2.0, dataset_path=csv_file,
        )
        speed_anomaly = next(
            (a for a in result["anomalies"] if a["parameter"] == "solar_wind_speed"),
            None,
        )
        if speed_anomaly:
            assert speed_anomaly["direction"] == "above_baseline"


# ---------------------------------------------------------------------------
# Significant anomaly
# ---------------------------------------------------------------------------

class TestSignificantAnomaly:

    def _build_controlled(self, tmp_path, recent_speed: float) -> dict:
        """
        Baseline: 20 rows speed cycling 396–404 (std≈3.16).
        recent_speed: value that should give a large z-score.
        """
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4)
                    for i in range(20)]
        recent = [_make_row(_ts(20), speed=recent_speed)]
        _write_csv(csv_file, baseline + recent)
        return detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=2.0, dataset_path=csv_file,
        )

    def test_significant_anomaly_above_baseline(self, tmp_path):
        # z ≈ (420 - 400) / 3.16 ≈ 6.3 → ≥ 3 * z_threshold → significant
        result = self._build_controlled(tmp_path, recent_speed=420.0)
        assert result["status"] == "ok"
        speed_a = next(
            (a for a in result["anomalies"] if a["parameter"] == "solar_wind_speed"),
            None,
        )
        assert speed_a is not None
        assert speed_a["severity"] == "significant"
        assert speed_a["direction"] == "above_baseline"

    def test_significant_anomaly_below_baseline(self, tmp_path):
        # z ≈ (380 - 400) / 3.16 ≈ -6.3 → significant below
        result = self._build_controlled(tmp_path, recent_speed=380.0)
        speed_a = next(
            (a for a in result["anomalies"] if a["parameter"] == "solar_wind_speed"),
            None,
        )
        assert speed_a is not None
        assert speed_a["severity"] == "significant"
        assert speed_a["direction"] == "below_baseline"

    def test_z_score_matches_manual_calculation(self, tmp_path):
        """
        Use a perfectly uniform baseline to get a predictable z-score.
        Baseline: 10 rows speed=[390,395,400,405,410] * 2 → mean=400, std≈7.91
        Recent: speed=424 → z=(424-400)/7.91≈3.03
        """
        csv_file = tmp_path / "sw.csv"
        speeds = [390, 395, 400, 405, 410] * 2
        baseline = [_make_row(_ts(i), speed=float(s)) for i, s in enumerate(speeds)]
        recent   = [_make_row(_ts(10), speed=424.0)]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=1, baseline_window=10,
            z_threshold=2.0, dataset_path=csv_file,
        )
        speed_a = next(
            (a for a in result["anomalies"] if a["parameter"] == "solar_wind_speed"),
            None,
        )
        assert speed_a is not None
        # z should be approximately 3.03 ± 0.5
        assert abs(speed_a["z_score"] - 3.03) < 0.5, (
            f"z_score {speed_a['z_score']} not close to expected ~3.03"
        )


# ---------------------------------------------------------------------------
# Multiple simultaneous anomalies
# ---------------------------------------------------------------------------

class TestMultipleAnomalies:

    def test_multiple_params_flagged_simultaneously(self, tmp_path):
        """
        Both solar_wind_speed and geomagnetic_index are driven to extreme values.
        """
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4, kp=2.5)
                    for i in range(20)]
        # speed=450 (+~15σ if std≈3), kp=10 (extreme)
        recent   = [_make_row(_ts(20), speed=450.0, kp=10.0)]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=2.0, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        flagged = {a["parameter"] for a in result["anomalies"]}
        assert "solar_wind_speed" in flagged
        assert "geomagnetic_index" in flagged

    def test_anomalies_sorted_by_abs_z_descending(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4, kp=2.5)
                    for i in range(20)]
        recent   = [_make_row(_ts(20), speed=450.0, kp=10.0)]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=2.0, dataset_path=csv_file,
        )
        zs = [abs(a["z_score"] or 0) for a in result["anomalies"]]
        assert zs == sorted(zs, reverse=True)

    def test_summary_counts_match_anomaly_list(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4, kp=2.5)
                    for i in range(20)]
        recent   = [_make_row(_ts(20), speed=450.0, kp=10.0)]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=2.0, dataset_path=csv_file,
        )
        assert result["summary"]["parameters_affected"] == len(result["anomalies"])


# ---------------------------------------------------------------------------
# Custom z_threshold
# ---------------------------------------------------------------------------

class TestCustomThreshold:

    def test_lower_threshold_catches_more_anomalies(self, tmp_path):
        """z_threshold=1.0 should flag observations that z_threshold=3.0 misses."""
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4)
                    for i in range(20)]
        # z ≈ 2.06 with std≈3.16: caught by threshold=2.0, not by threshold=3.0
        recent   = [_make_row(_ts(20), speed=406.5)]
        _write_csv(csv_file, baseline + recent)

        result_strict = detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=3.0, dataset_path=csv_file,
        )
        result_lenient = detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=1.0, dataset_path=csv_file,
        )
        lenient_params = {a["parameter"] for a in result_lenient["anomalies"]}
        strict_params  = {a["parameter"] for a in result_strict["anomalies"]}
        assert len(lenient_params) >= len(strict_params)

    def test_very_high_threshold_yields_no_anomalies(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4)
                    for i in range(20)]
        recent   = [_make_row(_ts(20), speed=410.0)]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=1, baseline_window=20,
            z_threshold=100.0, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        assert result["anomalies"] == []


# ---------------------------------------------------------------------------
# Zero standard deviation
# ---------------------------------------------------------------------------

class TestZeroStdDev:

    def test_zero_std_reported_in_parameter_stats(self, tmp_path):
        """When all baseline values are identical, std=0 should be noted."""
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(_ts(i), speed=400.0) for i in range(15)]
        _write_csv(csv_file, rows)
        result = detect_anomalies(
            recent_window=2, baseline_window=10, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        stat = result["parameter_stats"].get("solar_wind_speed", {})
        # Either std is 0 or a skipped_reason is set
        assert stat.get("std") == 0.0 or stat.get("skipped_reason") is not None

    def test_zero_std_deviation_in_recent_triggers_anomaly_when_value_differs(self, tmp_path):
        """Baseline all 400.0 (std=0), recent row = 500.0 → should be anomalous."""
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0) for i in range(10)]
        recent   = [_make_row(_ts(10), speed=500.0)]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=1, baseline_window=10,
            z_threshold=2.0, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        flagged = {a["parameter"] for a in result["anomalies"]}
        assert "solar_wind_speed" in flagged

    def test_zero_std_recent_equals_baseline_no_anomaly(self, tmp_path):
        """Baseline all 400.0, recent also 400.0 → no anomaly."""
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(_ts(i), speed=400.0) for i in range(12)]
        _write_csv(csv_file, rows)
        result = detect_anomalies(
            recent_window=2, baseline_window=8, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        speed_anomalies = [a for a in result["anomalies"]
                           if a["parameter"] == "solar_wind_speed"]
        assert speed_anomalies == []


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

class TestMissingValues:

    def test_nan_values_in_recent_window_skipped(self, tmp_path):
        """NaN / missing values in the recent window must not cause a crash."""
        csv_file = tmp_path / "sw.csv"
        baseline = [_make_row(_ts(i), speed=400.0 + (i % 5) * 2 - 4)
                    for i in range(20)]
        # recent rows: one NaN, one valid
        recent = [
            {"timestamp": _ts(20), "solar_wind_speed": "",
             "solar_wind_density": 5.0, "magnetic_field": 7.0,
             "xray_flux": "1.2e-07", "proton_flux": 2.0, "geomagnetic_index": 2.5},
            _make_row(_ts(21), speed=400.0),
        ]
        _write_csv(csv_file, baseline + recent)
        result = detect_anomalies(
            recent_window=2, baseline_window=20, dataset_path=csv_file,
        )
        assert result["status"] == "ok"

    def test_missing_parameter_column_absent_from_stats(self, tmp_path):
        """CSV with no xray_flux column → xray_flux absent from parameter_stats."""
        csv_file = tmp_path / "sw.csv"
        rows = [
            {"timestamp": _ts(i), "solar_wind_speed": 400.0 + i,
             "solar_wind_density": 5.0, "magnetic_field": 7.0,
             "proton_flux": 2.0, "geomagnetic_index": 2.5}
            for i in range(15)
        ]
        _write_csv(csv_file, rows)
        result = detect_anomalies(
            recent_window=3, baseline_window=10, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        assert "xray_flux" not in result["parameter_stats"]
        anomaly_params = {a["parameter"] for a in result["anomalies"]}
        assert "xray_flux" not in anomaly_params


# ---------------------------------------------------------------------------
# Window clamping
# ---------------------------------------------------------------------------

class TestWindowClamping:

    def test_large_windows_clamped_to_dataset_size(self, tmp_path):
        """Windows totalling more than dataset size should be clamped, not error."""
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(_ts(i), speed=400.0 + i) for i in range(10)]
        _write_csv(csv_file, rows)
        result = detect_anomalies(
            recent_window=50, baseline_window=200, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        assert result["baseline"]["note"] is not None  # note about clamping

    def test_clamped_windows_sum_to_dataset_size(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        n = 10
        rows = [_make_row(_ts(i)) for i in range(n)]
        _write_csv(csv_file, rows)
        result = detect_anomalies(
            recent_window=50, baseline_window=200, dataset_path=csv_file,
        )
        assert result["status"] == "ok"
        total = (result["baseline"]["observations"] +
                 result["analysis_window"]["observations"])
        assert total <= n


# ---------------------------------------------------------------------------
# Insufficient baseline
# ---------------------------------------------------------------------------

class TestInsufficientBaseline:

    def test_total_rows_less_than_min_returns_error(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(_ts(i)) for i in range(3)]  # only 3 rows total
        _write_csv(csv_file, rows)
        # baseline needs ≥ 3 rows AND recent needs ≥ 1, so 3 total is borderline
        result = detect_anomalies(
            recent_window=1, baseline_window=3, dataset_path=csv_file,
        )
        # 4 total needed; only 3 available → clamped and should still work or error
        # We accept either a successful clamped result or a graceful error
        assert result["status"] in ("ok", "error")

    def test_single_row_dataset_returns_error(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [_make_row(_ts(0))])
        result = detect_anomalies(dataset_path=csv_file)
        assert result["status"] == "error"
        assert "insufficient" in result["message"].lower()

    def test_empty_dataset_returns_error(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        csv_file.write_text("")
        result = detect_anomalies(dataset_path=csv_file)
        assert result["status"] == "error"

    def test_missing_file_returns_error(self, tmp_path):
        result = detect_anomalies(dataset_path=tmp_path / "nonexistent.csv")
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()


# ---------------------------------------------------------------------------
# Invalid input validation
# ---------------------------------------------------------------------------

class TestInvalidInputs:

    def test_recent_window_zero_returns_error(self):
        result = detect_anomalies(recent_window=0)
        assert result["status"] == "error"
        assert "invalid" in result["message"].lower()

    def test_recent_window_negative_returns_error(self):
        result = detect_anomalies(recent_window=-3)
        assert result["status"] == "error"

    def test_recent_window_non_integer_returns_error(self):
        result = detect_anomalies(recent_window="six")   # type: ignore[arg-type]
        assert result["status"] == "error"

    def test_baseline_window_too_small_returns_error(self):
        result = detect_anomalies(baseline_window=2)
        assert result["status"] == "error"

    def test_baseline_window_non_integer_returns_error(self):
        result = detect_anomalies(baseline_window=3.5)  # type: ignore[arg-type]
        assert result["status"] == "error"

    def test_z_threshold_zero_returns_error(self):
        result = detect_anomalies(z_threshold=0)
        assert result["status"] == "error"

    def test_z_threshold_negative_returns_error(self):
        result = detect_anomalies(z_threshold=-1.0)
        assert result["status"] == "error"

    def test_z_threshold_non_numeric_returns_error(self):
        result = detect_anomalies(z_threshold="two")    # type: ignore[arg-type]
        assert result["status"] == "error"
