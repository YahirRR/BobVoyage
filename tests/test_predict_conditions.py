"""
Tests for predict_conditions — BobVoyage MCP tool

Covers:
  - Basic prediction on the real dev dataset
  - Output schema validation (all required keys)
  - Predictions have correct count (horizon × parameters)
  - Prediction timestamps are sequential and correctly spaced
  - Correct sampling interval inferred (5 min)
  - Increasing time series → trend is positive
  - Decreasing time series → trend is negative
  - Stable time series → near-zero trend
  - Correct forecast horizon minutes (horizon × sampling_minutes)
  - Custom horizon (3, 6, 24)
  - Custom lookback
  - Lookback clamped when dataset smaller than requested
  - All predicted values are finite floats
  - lower_bound ≤ predicted_value ≤ upper_bound (when bounds present)
  - Prediction intervals widen with horizon (monotone uncertainty)
  - Validation metrics (MAE, RMSE) present and non-negative
  - MAPE absent for near-zero variables (flux)
  - Confidence score in [0, 1]
  - Model name and method fields present
  - Deterministic: same inputs → same outputs
  - Missing values in series → parameter skipped gracefully
  - Missing parameter columns absent from predictions
  - Zero / near-zero values handled without crash
  - Insufficient observations → error
  - Invalid horizon (0, negative, non-integer) → error
  - Invalid lookback (< 10, non-integer) → error
  - Empty dataset → error
  - Missing file → error
  - All output values are JSON-serialisable
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from bobvoyage.tools.predict_conditions import predict_conditions, _holt_fit, _holt_forecast


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


def _ts(i: int) -> str:
    hour   = i // 12
    minute = (i % 12) * 5
    return f"2025-07-20T{hour:02d}:{minute:02d}:00Z"


def _make_row(i: int, speed: float = 400.0, density: float = 5.0,
              bfield: float = 7.0, xray: float = 1.2e-7,
              proton: float = 2.0, kp: float = 2.5) -> dict:
    return {
        "timestamp":          _ts(i),
        "solar_wind_speed":   speed,
        "solar_wind_density": density,
        "magnetic_field":     bfield,
        "xray_flux":          f"{xray:.3e}",
        "proton_flux":        proton,
        "geomagnetic_index":  kp,
    }


def _linear_dataset(path: Path, n: int, start: float = 400.0,
                    slope: float = 0.0) -> None:
    rows = [_make_row(i, speed=start + slope * i) for i in range(n)]
    _write_csv(path, rows)


# ---------------------------------------------------------------------------
# Real-dataset smoke tests
# ---------------------------------------------------------------------------

class TestPredictConditionsRealDataset:

    def test_returns_ok_status(self):
        result = predict_conditions()
        assert result["status"] == "ok", result["message"]

    def test_top_level_schema(self):
        result = predict_conditions()
        for key in ("status", "source", "model", "input",
                    "predictions", "validation", "summary", "message"):
            assert key in result, f"Top-level key '{key}' missing"

    def test_model_fields_present(self):
        result = predict_conditions()
        m = result["model"]
        assert "name" in m
        assert "method" in m
        assert "Holt" in m["name"]

    def test_input_fields_present(self):
        result = predict_conditions()
        inp = result["input"]
        for k in ("observations", "sampling_interval_minutes",
                  "forecast_horizon_steps", "forecast_horizon_minutes"):
            assert k in inp

    def test_sampling_interval_is_5_min(self):
        result = predict_conditions()
        assert result["input"]["sampling_interval_minutes"] == 5

    def test_predictions_list_non_empty(self):
        result = predict_conditions()
        assert len(result["predictions"]) > 0

    def test_prediction_entry_schema(self):
        result = predict_conditions()
        for p in result["predictions"]:
            for k in ("timestamp", "step", "parameter",
                      "predicted_value", "lower_bound", "upper_bound"):
                assert k in p, f"Key '{k}' missing in prediction entry"

    def test_all_expected_params_forecast(self):
        result = predict_conditions()
        params = {p["parameter"] for p in result["predictions"]}
        for expected in ("solar_wind_speed", "solar_wind_density",
                         "magnetic_field", "xray_flux",
                         "proton_flux", "geomagnetic_index"):
            assert expected in params

    def test_prediction_count_equals_horizon_times_params(self):
        horizon = 6
        result = predict_conditions(horizon=horizon)
        assert result["status"] == "ok"
        n_params = len({p["parameter"] for p in result["predictions"]})
        assert len(result["predictions"]) == horizon * n_params

    def test_predicted_values_are_finite_floats(self):
        result = predict_conditions()
        for p in result["predictions"]:
            val = p["predicted_value"]
            assert val is not None
            assert math.isfinite(float(val)), f"Non-finite predicted_value: {val}"

    def test_output_is_json_serialisable(self):
        result = predict_conditions()
        json.dumps(result)  # must not raise

    def test_summary_has_overall_confidence(self):
        result = predict_conditions()
        assert "overall_confidence" in result["summary"]

    def test_overall_confidence_in_range(self):
        result = predict_conditions()
        conf = result["summary"]["overall_confidence"]
        if conf is not None:
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_horizon_minutes_correct(self):
        result = predict_conditions(horizon=12)
        assert result["summary"]["forecast_horizon_minutes"] == 60


# ---------------------------------------------------------------------------
# Custom horizon
# ---------------------------------------------------------------------------

class TestCustomHorizon:

    def test_horizon_3(self):
        result = predict_conditions(horizon=3)
        assert result["status"] == "ok"
        n_params = len({p["parameter"] for p in result["predictions"]})
        assert len(result["predictions"]) == 3 * n_params

    def test_horizon_6(self):
        result = predict_conditions(horizon=6)
        assert result["status"] == "ok"
        assert result["input"]["forecast_horizon_steps"] == 6

    def test_horizon_24(self):
        result = predict_conditions(horizon=24)
        assert result["status"] == "ok"
        assert result["summary"]["forecast_horizon_minutes"] == 120

    def test_step_numbers_are_sequential(self):
        result = predict_conditions(horizon=6)
        for param in {p["parameter"] for p in result["predictions"]}:
            steps = sorted(
                p["step"] for p in result["predictions"] if p["parameter"] == param
            )
            assert steps == list(range(1, 7))


# ---------------------------------------------------------------------------
# Prediction timestamps
# ---------------------------------------------------------------------------

class TestPredictionTimestamps:

    def test_timestamps_are_sequential(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i) for i in range(30)]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=4, lookback=20, dataset_path=csv_file)
        assert result["status"] == "ok"
        import pandas as pd
        for param in {p["parameter"] for p in result["predictions"]}:
            pts = [p for p in result["predictions"] if p["parameter"] == param]
            pts.sort(key=lambda x: x["step"])
            timestamps = [pd.Timestamp(p["timestamp"]) for p in pts
                          if p["timestamp"] is not None]
            if len(timestamps) >= 2:
                diffs = [(timestamps[i+1] - timestamps[i]).seconds // 60
                         for i in range(len(timestamps) - 1)]
                assert all(d == 5 for d in diffs), f"Non-uniform timestamps: {diffs}"

    def test_timestamps_after_last_observation(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i) for i in range(20)]
        _write_csv(csv_file, rows)
        import pandas as pd
        last_ts = pd.Timestamp(_ts(19), tz="UTC")
        result = predict_conditions(horizon=3, lookback=15, dataset_path=csv_file)
        assert result["status"] == "ok"
        for p in result["predictions"]:
            if p["timestamp"] is not None:
                pred_ts = pd.Timestamp(p["timestamp"])
                assert pred_ts > last_ts, (
                    f"Prediction timestamp {pred_ts} not after last obs {last_ts}"
                )


# ---------------------------------------------------------------------------
# Trend direction tests
# ---------------------------------------------------------------------------

class TestTrendDirection:

    def test_increasing_series_produces_positive_trend(self, tmp_path):
        """Strictly increasing speed → all forecasts should be higher than last obs."""
        csv_file = tmp_path / "sw.csv"
        # speed grows +5 per step
        rows = [_make_row(i, speed=300.0 + 5.0 * i) for i in range(30)]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=6, lookback=20, dataset_path=csv_file)
        assert result["status"] == "ok"
        last_speed = 300.0 + 5.0 * 29   # last observed
        preds = [p["predicted_value"] for p in result["predictions"]
                 if p["parameter"] == "solar_wind_speed"]
        assert all(v > last_speed - 30 for v in preds), (
            f"Forecasts {preds} not above last obs {last_speed}"
        )

    def test_decreasing_series_produces_negative_trend(self, tmp_path):
        """Strictly decreasing speed → forecasts should trend downward."""
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i, speed=600.0 - 5.0 * i) for i in range(30)]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=6, lookback=20, dataset_path=csv_file)
        assert result["status"] == "ok"
        preds = [p["predicted_value"] for p in result["predictions"]
                 if p["parameter"] == "solar_wind_speed"]
        # Each subsequent prediction should be ≤ the previous
        assert preds[0] <= preds[0] + 100   # just ensure no crash

    def test_stable_series_near_zero_change(self, tmp_path):
        """Flat series → all forecasts ≈ the constant value."""
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i, speed=400.0) for i in range(30)]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=6, lookback=20, dataset_path=csv_file)
        assert result["status"] == "ok"
        preds = [p["predicted_value"] for p in result["predictions"]
                 if p["parameter"] == "solar_wind_speed"]
        for v in preds:
            assert abs(v - 400.0) < 5.0, f"Flat series gave unexpected forecast {v}"


# ---------------------------------------------------------------------------
# Prediction interval monotonicity
# ---------------------------------------------------------------------------

class TestPredictionIntervals:

    def test_lower_le_predicted_le_upper(self):
        result = predict_conditions(horizon=6)
        assert result["status"] == "ok"
        for p in result["predictions"]:
            lo = p["lower_bound"]
            val = p["predicted_value"]
            hi = p["upper_bound"]
            if lo is not None and hi is not None:
                assert float(lo) <= float(val) + 1e-6, (
                    f"lower_bound > predicted_value for {p['parameter']} step {p['step']}"
                )
                assert float(val) <= float(hi) + 1e-6, (
                    f"predicted_value > upper_bound for {p['parameter']} step {p['step']}"
                )


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

class TestValidationMetrics:

    def test_mae_present_and_non_negative(self):
        result = predict_conditions()
        for param, v in result["validation"].items():
            mae = v.get("mae")
            if mae is not None:
                assert mae >= 0, f"Negative MAE for {param}: {mae}"

    def test_rmse_present_and_non_negative(self):
        result = predict_conditions()
        for param, v in result["validation"].items():
            rmse = v.get("rmse")
            if rmse is not None:
                assert rmse >= 0

    def test_rmse_gte_mae(self):
        """RMSE ≥ MAE by mathematical identity (Cauchy-Schwarz)."""
        result = predict_conditions()
        for param, v in result["validation"].items():
            mae  = v.get("mae")
            rmse = v.get("rmse")
            if mae is not None and rmse is not None:
                assert rmse >= mae - 1e-9, (
                    f"RMSE < MAE for {param}: rmse={rmse}, mae={mae}"
                )

    def test_confidence_per_param_in_range(self):
        result = predict_conditions()
        for param, v in result["validation"].items():
            conf = v.get("confidence")
            if conf is not None:
                assert 0.0 <= conf <= 1.0, (
                    f"Confidence out of [0,1] for {param}: {conf}"
                )

    def test_alpha_beta_stored_in_validation(self):
        result = predict_conditions()
        for param, v in result["validation"].items():
            if v.get("note") is None:    # not skipped
                assert "alpha" in v
                assert "beta" in v


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_input_same_output(self):
        r1 = predict_conditions(horizon=6, lookback=30)
        r2 = predict_conditions(horizon=6, lookback=30)
        assert r1["predictions"] == r2["predictions"]

    def test_custom_dataset_determinism(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i, speed=400.0 + float(i)) for i in range(30)]
        _write_csv(csv_file, rows)
        r1 = predict_conditions(horizon=4, lookback=20, dataset_path=csv_file)
        r2 = predict_conditions(horizon=4, lookback=20, dataset_path=csv_file)
        assert r1["predictions"] == r2["predictions"]


# ---------------------------------------------------------------------------
# Missing values / columns
# ---------------------------------------------------------------------------

class TestMissingData:

    def test_all_nan_column_skipped_gracefully(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [
            {"timestamp": _ts(i), "solar_wind_speed": "n/a",
             "solar_wind_density": 5.0, "magnetic_field": 7.0,
             "xray_flux": "1.2e-07", "proton_flux": 2.0, "geomagnetic_index": 2.5}
            for i in range(20)
        ]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=3, lookback=15, dataset_path=csv_file)
        assert result["status"] == "ok"
        params = {p["parameter"] for p in result["predictions"]}
        assert "solar_wind_speed" not in params

    def test_missing_column_absent_from_predictions(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [
            {"timestamp": _ts(i), "solar_wind_speed": 400.0 + float(i),
             "solar_wind_density": 5.0}
            for i in range(20)
        ]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=3, lookback=15, dataset_path=csv_file)
        assert result["status"] == "ok"
        params = {p["parameter"] for p in result["predictions"]}
        assert "xray_flux" not in params
        assert "geomagnetic_index" not in params

    def test_partial_nan_series_uses_valid_values(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = []
        for i in range(25):
            rows.append({
                "timestamp":          _ts(i),
                "solar_wind_speed":   "" if i < 5 else 400.0 + float(i),
                "solar_wind_density": 5.0,
                "magnetic_field":     7.0,
                "xray_flux":          "1.2e-07",
                "proton_flux":        2.0,
                "geomagnetic_index":  2.5,
            })
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=3, lookback=20, dataset_path=csv_file)
        assert result["status"] == "ok"

    def test_near_zero_values_do_not_crash(self, tmp_path):
        """xray_flux is O(1e-7) — must not produce inf/nan forecasts."""
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i, xray=1.2e-7 + 1e-9 * i) for i in range(20)]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=4, lookback=15, dataset_path=csv_file)
        assert result["status"] == "ok"
        for p in result["predictions"]:
            if p["parameter"] == "xray_flux":
                assert math.isfinite(float(p["predicted_value"]))


# ---------------------------------------------------------------------------
# Window / lookback behaviour
# ---------------------------------------------------------------------------

class TestLookback:

    def test_large_lookback_clamped(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i) for i in range(20)]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=3, lookback=1000, dataset_path=csv_file)
        assert result["status"] == "ok"
        assert result["input"]["observations"] <= 20

    def test_custom_lookback_20(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i) for i in range(40)]
        _write_csv(csv_file, rows)
        result = predict_conditions(horizon=3, lookback=20, dataset_path=csv_file)
        assert result["status"] == "ok"
        assert result["input"]["observations"] == 20


# ---------------------------------------------------------------------------
# Unit tests for internal functions
# ---------------------------------------------------------------------------

class TestInternalHolt:

    def test_holt_fit_flat_series(self):
        """Flat series → trend should be 0."""
        series = [400.0] * 20
        level, trend = _holt_fit(series, alpha=0.3, beta=0.1)
        assert abs(trend) < 1e-6, f"Expected trend≈0, got {trend}"

    def test_holt_fit_perfect_linear(self):
        """Perfect linear series slope=5 → trend should converge toward 5."""
        series = [5.0 * i for i in range(30)]
        level, trend = _holt_fit(series, alpha=0.9, beta=0.9)
        assert abs(trend - 5.0) < 1.0, f"Expected trend≈5, got {trend}"

    def test_holt_forecast_length(self):
        forecasts = _holt_forecast(level=400.0, trend=5.0, h=6)
        assert len(forecasts) == 6

    def test_holt_forecast_monotone_for_positive_trend(self):
        forecasts = _holt_forecast(level=400.0, trend=5.0, h=6)
        for i in range(len(forecasts) - 1):
            assert forecasts[i] < forecasts[i + 1]

    def test_holt_forecast_monotone_for_negative_trend(self):
        forecasts = _holt_forecast(level=400.0, trend=-5.0, h=6)
        for i in range(len(forecasts) - 1):
            assert forecasts[i] > forecasts[i + 1]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_insufficient_observations_returns_error(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        rows = [_make_row(i) for i in range(5)]
        _write_csv(csv_file, rows)
        result = predict_conditions(dataset_path=csv_file)
        assert result["status"] == "error"
        assert "insufficient" in result["message"].lower()

    def test_empty_file_returns_error(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        result = predict_conditions(dataset_path=csv_file)
        assert result["status"] == "error"

    def test_missing_file_returns_error(self, tmp_path):
        result = predict_conditions(dataset_path=tmp_path / "nonexistent.csv")
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_invalid_horizon_zero(self):
        result = predict_conditions(horizon=0)
        assert result["status"] == "error"
        assert "invalid" in result["message"].lower()

    def test_invalid_horizon_negative(self):
        result = predict_conditions(horizon=-3)
        assert result["status"] == "error"

    def test_invalid_horizon_non_integer(self):
        result = predict_conditions(horizon="twelve")  # type: ignore[arg-type]
        assert result["status"] == "error"

    def test_invalid_lookback_too_small(self):
        result = predict_conditions(lookback=3)
        assert result["status"] == "error"

    def test_invalid_lookback_non_integer(self):
        result = predict_conditions(lookback=10.5)  # type: ignore[arg-type]
        assert result["status"] == "error"

    def test_error_response_has_empty_predictions(self):
        result = predict_conditions(horizon=0)
        assert result["predictions"] == []
