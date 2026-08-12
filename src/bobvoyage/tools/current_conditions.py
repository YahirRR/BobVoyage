"""
get_current_conditions — BobVoyage MCP tool

Retrieves the most recent space-weather observation from the local CSV dataset.
Returns a structured dict with all available measurement fields.

Responsibility: data retrieval ONLY.
No prediction, anomaly detection, or risk assessment is performed here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset location — resolved relative to the project root so the tool works
# regardless of the current working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/bobvoyage/tools -> root
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "space_weather.csv"

# Expected column names in the CSV
_EXPECTED_COLUMNS = {
    "timestamp",
    "solar_wind_speed",
    "solar_wind_density",
    "magnetic_field",
    "xray_flux",
    "proton_flux",
    "geomagnetic_index",
}


def get_current_conditions(dataset_path: str | Path | None = None) -> dict[str, Any]:
    """Return the most recent space-weather observation from the local dataset.

    Parameters
    ----------
    dataset_path:
        Optional override for the CSV file location.
        Defaults to ``data/space_weather.csv`` at the project root.

    Returns
    -------
    dict with keys:
        status          – "ok" | "error"
        source          – absolute path of the dataset used
        observation     – dict of measurement fields (None when unavailable)
        missing_fields  – list of expected fields absent from the dataset
        message         – human-readable status message
    """
    path = Path(dataset_path) if dataset_path else _DEFAULT_DATASET

    # --- validate file exists -------------------------------------------------
    if not path.exists():
        return {
            "status": "error",
            "source": str(path),
            "observation": None,
            "missing_fields": [],
            "message": f"Dataset not found at '{path}'.",
        }

    # --- load CSV -------------------------------------------------------------
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "source": str(path),
            "observation": None,
            "missing_fields": [],
            "message": f"Failed to read dataset: {exc}",
        }

    if df.empty:
        return {
            "status": "error",
            "source": str(path),
            "observation": None,
            "missing_fields": [],
            "message": "Dataset is empty — no observations available.",
        }

    # --- identify missing fields ----------------------------------------------
    available_columns = set(df.columns)
    missing_fields = sorted(_EXPECTED_COLUMNS - available_columns)

    # --- select the latest row ------------------------------------------------
    if "timestamp" in available_columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return {
                "status": "error",
                "source": str(path),
                "observation": None,
                "missing_fields": missing_fields,
                "message": "No rows with a valid timestamp found in the dataset.",
            }
        latest_row = df.sort_values("timestamp").iloc[-1]
    else:
        latest_row = df.iloc[-1]

    # --- build observation dict -----------------------------------------------
    observation: dict[str, Any] = {}

    # timestamp — serialise to ISO-8601 string
    if "timestamp" in available_columns:
        ts = latest_row["timestamp"]
        observation["timestamp"] = ts.isoformat() if pd.notna(ts) else None
    else:
        observation["timestamp"] = None

    # numeric fields — convert NaN → None so the result is JSON-serialisable
    for field in (
        "solar_wind_speed",
        "solar_wind_density",
        "magnetic_field",
        "xray_flux",
        "proton_flux",
        "geomagnetic_index",
    ):
        if field in available_columns:
            raw = latest_row[field]
            if pd.isna(raw):
                observation[field] = None
            else:
                # xray_flux may be stored as a string in scientific notation
                try:
                    observation[field] = float(raw)
                except (ValueError, TypeError):
                    observation[field] = str(raw)
        else:
            observation[field] = None

    return {
        "status": "ok",
        "source": str(path),
        "observation": observation,
        "missing_fields": missing_fields,
        "message": "Most recent observation retrieved successfully.",
    }
