"""
predict_conditions — BobVoyage MCP tool

Generates short-term space-weather forecasts using Holt's Double Exponential
Smoothing (linear trend model).

===== MODEL SELECTION RATIONALE =====

Dataset characteristics (200 rows, 5-min interval, 16.6 h span):

  lag-1 autocorrelation:  0.92 – 0.99  → strong serial correlation
  first-difference mean:  ≈ 0           → no persistent drift
  first-difference std:   small vs mean → low-noise smooth series
  Missing values:         none
  Series length:          200 pts

Conclusion: all series are smooth, strongly autocorrelated, and nearly
stationary after first differencing.  This is the textbook case for
exponential smoothing.

Why Holt's Double Exponential Smoothing (DES)?
  • Captures both the level and the local linear trend at each step —
    superior to simple exponential smoothing for series with short-term
    drift.
  • Single formula, no matrix operations, O(n) fit — runs in milliseconds.
  • Fully explainable: two interpretable parameters (α, β).
  • No risk of overfitting on 200 points.
  • Appropriate for 1–24 step horizons (5 – 120 min at 5-min resolution).

Why NOT more complex models?
  • ARIMA: requires stationarity testing + order selection; overkill for
    200 smooth, nearly-stationary observations with no seasonal period.
  • Prophet / NeuralProphet: designed for daily/weekly seasonality;
    unnecessary complexity and dependency weight for a 16-hour dataset.
  • LSTM: requires far more data; would overfit badly on 200 rows.

Validation strategy:
  • Time-aware rolling origin: last `validation_steps` observations are
    withheld; the model is fit on everything before them and evaluated
    one-step-ahead.  No shuffling — temporal order is preserved.
  • Metrics: MAE and RMSE.  MAPE is computed only for parameters whose
    mean absolute value exceeds a safety floor (avoids division-by-~0
    for flux variables).

Responsibility: forecasting ONLY.
No anomaly detection, trend labelling, or risk assessment here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "space_weather.csv"

_NUMERIC_PARAMS: list[str] = [
    "solar_wind_speed",
    "solar_wind_density",
    "magnetic_field",
    "xray_flux",
    "proton_flux",
    "geomagnetic_index",
]

# MAPE is undefined / misleading when values approach zero
_MAPE_MIN_MEAN = 1e-6

# Minimum observations required: enough to fit + validate
_MIN_TRAIN = 10
_VALIDATION_STEPS = 6   # withheld for walk-forward validation


# ---------------------------------------------------------------------------
# Holt Double Exponential Smoothing
# ---------------------------------------------------------------------------

def _holt_fit(series: list[float], alpha: float, beta: float) -> tuple[float, float]:
    """Fit Holt's DES on `series` and return (level, trend) at last point."""
    if len(series) < 2:
        raise ValueError("Need at least 2 observations to fit Holt DES.")
    # Initialise: level = first value, trend = second − first
    level = series[0]
    trend = series[1] - series[0]
    for val in series[1:]:
        prev_level = level
        level = alpha * val + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return level, trend


def _holt_forecast(level: float, trend: float, h: int) -> list[float]:
    """Generate h-step-ahead forecasts from fitted level and trend."""
    return [level + (i + 1) * trend for i in range(h)]


def _grid_search_alpha_beta(
    series: list[float],
    step: float = 0.1,
) -> tuple[float, float]:
    """
    Brute-force grid search for (alpha, beta) minimising one-step-ahead
    in-sample SSE.  Coarse grid (step=0.1) is fast and sufficient for
    smooth space-weather series.
    """
    best_sse = math.inf
    best_alpha, best_beta = 0.3, 0.1

    candidates = [round(v * step, 10) for v in range(1, int(1 / step))]

    for alpha in candidates:
        for beta in candidates:
            sse = 0.0
            level = series[0]
            trend = series[1] - series[0]
            for val in series[1:]:
                pred = level + trend
                sse += (val - pred) ** 2
                prev_level = level
                level = alpha * val + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
            if sse < best_sse:
                best_sse = sse
                best_alpha, best_beta = alpha, beta

    return best_alpha, best_beta


# ---------------------------------------------------------------------------
# Validation (rolling-origin / walk-forward)
# ---------------------------------------------------------------------------

def _validate(
    series: list[float],
    alpha: float,
    beta: float,
    steps: int,
) -> dict[str, float | None]:
    """
    Walk-forward validation: for each of the last `steps` observations,
    fit on everything before it and forecast 1-step-ahead.
    Returns MAE, RMSE, MAPE (where applicable).
    """
    n = len(series)
    if n < steps + _MIN_TRAIN:
        return {"mae": None, "rmse": None, "mape": None,
                "note": "Insufficient data for walk-forward validation."}

    errors: list[float] = []
    abs_actuals: list[float] = []

    for i in range(steps):
        cutoff = n - steps + i          # train on series[:cutoff]
        train  = series[:cutoff]
        actual = series[cutoff]
        level, trend = _holt_fit(train, alpha, beta)
        predicted = level + trend       # 1-step-ahead
        errors.append(predicted - actual)
        abs_actuals.append(abs(actual))

    abs_errors = [abs(e) for e in errors]
    mae  = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))

    mean_actual = sum(abs_actuals) / len(abs_actuals)
    if mean_actual > _MAPE_MIN_MEAN:
        mape: float | None = (
            sum(abs(e) / max(abs(a), 1e-30)
                for e, a in zip(errors, abs_actuals)) / len(errors) * 100.0
        )
    else:
        mape = None

    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "mape": mape,
            "note": None}


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def _confidence_from_rmse(rmse: float | None, mean_val: float) -> float | None:
    """
    Derive a [0, 1] confidence score from normalised RMSE.

    confidence = max(0, 1 - nRMSE)
    nRMSE = RMSE / |mean|

    This is a heuristic, not a formal statistical interval, but it is
    explainable and monotonically penalises larger errors.
    """
    if rmse is None or mean_val == 0:
        return None
    nrmse = rmse / abs(mean_val)
    return round(max(0.0, min(1.0, 1.0 - nrmse)), 3)


# ---------------------------------------------------------------------------
# Safe float
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _fmt(val: float | None, decimals: int = 4) -> float | None:
    if val is None:
        return None
    if val != 0 and abs(val) < 1e-3:
        return val          # preserve precision for scientific quantities
    return round(val, decimals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_conditions(
    horizon: int = 12,
    lookback: int = 48,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Forecast space-weather parameters using Holt's Double Exponential Smoothing.

    Parameters
    ----------
    horizon:
        Number of future steps to forecast.  Must be ≥ 1.
        At 5-min resolution: horizon=12 → 60 min, horizon=6 → 30 min.
    lookback:
        Number of most-recent historical observations used for fitting.
        Must be ≥ 10.  Defaults to 48 (≈ 4 hours).
    dataset_path:
        Optional CSV path override.
        Defaults to ``data/space_weather.csv`` at the project root.

    Returns
    -------
    dict with keys:
        status             – "ok" | "error"
        source             – dataset path used
        model              – {name, method, alpha, beta, note}
        input              – {observations, sampling_interval_minutes,
                               forecast_horizon_steps,
                               forecast_horizon_minutes}
        predictions        – list of {timestamp, parameter,
                               predicted_value, lower_bound, upper_bound}
        validation         – per-parameter {mae, rmse, mape, confidence, note}
        summary            – {forecast_horizon_minutes, parameters_forecast,
                               overall_confidence}
        message            – human-readable status
    """
    # --- input validation ----------------------------------------------------
    errors: list[str] = []
    if not isinstance(horizon, int) or horizon < 1:
        errors.append(f"horizon must be an integer ≥ 1 (got {horizon!r}).")
    if not isinstance(lookback, int) or lookback < _MIN_TRAIN:
        errors.append(
            f"lookback must be an integer ≥ {_MIN_TRAIN} (got {lookback!r})."
        )
    if errors:
        return _error(None, "Invalid parameters: " + " ".join(errors))

    path = Path(dataset_path) if dataset_path else _DEFAULT_DATASET

    # --- load -----------------------------------------------------------------
    if not path.exists():
        return _error(str(path), f"Dataset not found at '{path}'.")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return _error(str(path), f"Failed to read dataset: {exc}.")

    if df.empty:
        return _error(str(path), "Dataset is empty.")

    # --- sort + infer sampling interval --------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    n_total = len(df)
    if n_total < _MIN_TRAIN + 1:
        return _error(
            str(path),
            f"Insufficient data: {n_total} rows; at least {_MIN_TRAIN + 1} required.",
        )

    # Infer sampling interval from median delta
    if "timestamp" in df.columns and len(df) > 1:
        deltas = df["timestamp"].diff().dropna()
        median_delta = deltas.median()
        sampling_minutes = int(round(median_delta.total_seconds() / 60))
    else:
        sampling_minutes = 5    # sensible default for this dataset

    # Clamp lookback
    eff_lookback = min(lookback, n_total)
    train_df = df.tail(eff_lookback).reset_index(drop=True)
    last_ts = train_df["timestamp"].iloc[-1] if "timestamp" in train_df.columns else None

    # --- per-parameter fit + forecast -----------------------------------------
    all_predictions: list[dict[str, Any]] = []
    validation_results: dict[str, Any] = {}
    confidences: list[float] = []

    for param in _NUMERIC_PARAMS:
        if param not in train_df.columns:
            continue

        series_raw = pd.to_numeric(train_df[param], errors="coerce")
        series = [v for v in series_raw if not (math.isnan(v) or math.isinf(v))]

        if len(series) < _MIN_TRAIN:
            validation_results[param] = {
                "mae": None, "rmse": None, "mape": None, "confidence": None,
                "note": f"Only {len(series)} valid values; skipped.",
            }
            continue

        # grid-search optimal alpha, beta
        alpha, beta = _grid_search_alpha_beta(series)

        # validate before committing to forecast
        val_metrics = _validate(series, alpha, beta, steps=min(_VALIDATION_STEPS, len(series) // 3))
        mean_val = sum(abs(v) for v in series) / len(series)
        conf = _confidence_from_rmse(val_metrics["mae"], mean_val)

        validation_results[param] = {
            "mae":        _fmt(val_metrics["mae"]),
            "rmse":       _fmt(val_metrics["rmse"]),
            "mape":       _fmt(val_metrics["mape"], 2) if val_metrics["mape"] is not None else None,
            "confidence": conf,
            "alpha":      alpha,
            "beta":       beta,
            "note":       val_metrics.get("note"),
        }
        if conf is not None:
            confidences.append(conf)

        # fit on full training series
        level, trend = _holt_fit(series, alpha, beta)
        forecasts = _holt_forecast(level, trend, horizon)

        # prediction interval: ±1.96 * MAE (Gaussian approximation)
        pi_half: float | None = None
        if val_metrics["mae"] is not None:
            pi_half = 1.96 * val_metrics["mae"]

        for h_step, pred_val in enumerate(forecasts, start=1):
            # timestamp
            if last_ts is not None and pd.notna(last_ts):
                step_ts = last_ts + pd.Timedelta(minutes=sampling_minutes * h_step)
                ts_str: str | None = step_ts.isoformat()
            else:
                ts_str = None

            lower: float | None = None
            upper: float | None = None
            if pi_half is not None:
                lower = _fmt(pred_val - pi_half)
                upper = _fmt(pred_val + pi_half)

            all_predictions.append({
                "timestamp":       ts_str,
                "step":            h_step,
                "parameter":       param,
                "predicted_value": _fmt(pred_val),
                "lower_bound":     lower,
                "upper_bound":     upper,
            })

    if not all_predictions:
        return _error(
            str(path),
            "No parameters could be forecast from the available data.",
        )

    # --- overall confidence --------------------------------------------------
    overall_conf: float | None = (
        round(sum(confidences) / len(confidences), 3) if confidences else None
    )

    # --- summary -------------------------------------------------------------
    params_forecast = sorted({p["parameter"] for p in all_predictions})
    summary = {
        "forecast_horizon_steps":   horizon,
        "forecast_horizon_minutes": horizon * sampling_minutes,
        "sampling_interval_minutes": sampling_minutes,
        "parameters_forecast":      params_forecast,
        "overall_confidence":       overall_conf,
    }

    return {
        "status":  "ok",
        "source":  str(path),
        "model": {
            "name":   "Holt Double Exponential Smoothing",
            "method": (
                "Two-parameter exponential smoothing with local linear trend. "
                "alpha (level smoothing) and beta (trend smoothing) are selected "
                "via in-sample SSE minimisation. Validated with walk-forward "
                "one-step-ahead evaluation."
            ),
        },
        "input": {
            "observations":              eff_lookback,
            "sampling_interval_minutes": sampling_minutes,
            "forecast_horizon_steps":    horizon,
            "forecast_horizon_minutes":  horizon * sampling_minutes,
        },
        "predictions": all_predictions,
        "validation":  validation_results,
        "summary":     summary,
        "message": (
            f"Forecast generated for {len(params_forecast)} parameter(s) "
            f"over {horizon} step(s) "
            f"({horizon * sampling_minutes} min) "
            f"using Holt Double Exponential Smoothing."
        ),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _error(source: str | None, message: str) -> dict[str, Any]:
    return {
        "status":      "error",
        "source":      source,
        "model":       None,
        "input":       None,
        "predictions": [],
        "validation":  {},
        "summary":     {
            "forecast_horizon_steps":    None,
            "forecast_horizon_minutes":  None,
            "sampling_interval_minutes": None,
            "parameters_forecast":       [],
            "overall_confidence":        None,
        },
        "message": message,
    }
