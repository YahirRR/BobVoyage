"""
get_current_conditions — BobVoyage MCP tool

Retrieves the most recent space-weather observation.

Provider selection:
  • When ``dataset_path`` is explicitly supplied (or BOBVOYAGE_DATA_PROVIDER
    is "local" / unset), the original CSV path is used directly — this keeps
    all existing tests deterministic with zero network dependency.
  • When BOBVOYAGE_DATA_PROVIDER is set to "noaa" or another live provider,
    the request is delegated to the provider layer and the canonical
    SpaceWeatherObservation is returned in the same dict schema.

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

_PROVIDER_ENV = "BOBVOYAGE_DATA_PROVIDER"


def _use_provider_layer(dataset_path: str | Path | None) -> bool:
    """Return True when the request should be routed through the provider layer."""
    if dataset_path is not None:
        return False   # explicit path → always use CSV directly
    active = os.environ.get(_PROVIDER_ENV, "local").lower().strip()
    return active not in ("local", "")


def _from_provider() -> dict[str, Any]:
    """Delegate to the active provider and normalise the response to the tool schema."""
    from bobvoyage.data.factory import get_provider
    provider = get_provider()
    resp     = provider.get_current_conditions()

    if resp.status == "error":
        return {
            "status":        "error",
            "source":        resp.source,
            "observation":   None,
            "missing_fields": [],
            "data_age_seconds": None,
            "is_stale":      False,
            "message":       resp.message,
        }

    obs = resp.observation
    observation: dict[str, Any] = obs.to_dict() if obs else {}

    # Determine which expected fields are absent from the provider response
    missing = sorted(
        f for f in _EXPECTED_COLUMNS
        if observation.get(f) is None and f != "timestamp"
    )

    return {
        "status":           resp.status,
        "source":           resp.source,
        "observation":      observation,
        "missing_fields":   missing,
        "data_age_seconds": obs.data_age_seconds if obs else None,
        "is_stale":         obs.is_stale          if obs else False,
        "message":          resp.message,
    }


def get_current_conditions(dataset_path: str | Path | None = None) -> dict[str, Any]:
    """Return the most recent space-weather observation.

    Parameters
    ----------
    dataset_path:
        Optional override for the CSV file location.
        Defaults to ``data/space_weather.csv`` at the project root.
        When set, always reads from the local CSV regardless of
        BOBVOYAGE_DATA_PROVIDER.

    Returns
    -------
    dict with keys:
        status            – "ok" | "degraded" | "error"
        source            – path or provider name
        observation       – dict of measurement fields (None when unavailable)
        missing_fields    – list of expected fields absent from the observation
        data_age_seconds  – age of the observation in seconds (live providers)
        is_stale          – True if observation exceeds freshness threshold
        message           – human-readable status message
    """
    # --- route to provider layer if configured --------------------------------
    if _use_provider_layer(dataset_path):
        return _from_provider()

    # --- original CSV path -------------------------------------------------------
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
        "status":           "ok",
        "source":           str(path),
        "observation":      observation,
        "missing_fields":   missing_fields,
        "data_age_seconds": None,
        "is_stale":         False,
        "message":          "Most recent observation retrieved successfully.",
    }
