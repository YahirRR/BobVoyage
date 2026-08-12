"""
LocalProvider — reads space-weather data from the local CSV dataset.

This is the deterministic development and test provider.
It must always remain available and must never call external APIs.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bobvoyage.data.models.space_weather import (
    ProviderResponse,
    SpaceWeatherObservation,
    SpaceWeatherEvent,
)
from bobvoyage.data.providers.base import SpaceWeatherProvider

_PROJECT_ROOT  = Path(__file__).resolve().parents[4]
_DEFAULT_CSV   = _PROJECT_ROOT / "data" / "space_weather.csv"


class LocalProvider(SpaceWeatherProvider):
    """Reads observations from a local CSV file."""

    SOURCE_NAME              = "LOCAL"
    STALE_THRESHOLD_SECONDS  = float("inf")  # local data is never "stale"

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load_sorted(self) -> pd.DataFrame:
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Local CSV not found: '{self._csv_path}'")
        df = pd.read_csv(self._csv_path)
        if df.empty:
            raise ValueError("Local CSV is empty.")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def _row_to_obs(self, row: pd.Series, retrieved_at: str) -> SpaceWeatherObservation:
        sf = self._safe_float
        ts = row.get("timestamp")
        if hasattr(ts, "isoformat"):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts) if ts is not None and str(ts) != "NaT" else None

        return SpaceWeatherObservation(
            timestamp          = ts_str,
            solar_wind_speed   = sf(row.get("solar_wind_speed")),
            solar_wind_density = sf(row.get("solar_wind_density")),
            magnetic_field     = sf(row.get("magnetic_field")),
            xray_flux          = sf(row.get("xray_flux")),
            proton_flux        = sf(row.get("proton_flux")),
            geomagnetic_index  = sf(row.get("geomagnetic_index")),
            source             = self.SOURCE_NAME,
            retrieved_at       = retrieved_at,
            data_age_seconds   = None,   # local data has no meaningful age
            is_stale           = False,
        )

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def get_current_conditions(self) -> ProviderResponse:
        retrieved_at = self._now_utc_iso()
        try:
            df  = self._load_sorted()
            obs = self._row_to_obs(df.iloc[-1], retrieved_at)
            return ProviderResponse(
                status      = "ok",
                source      = self.SOURCE_NAME,
                observation = obs,
                message     = "Most recent local observation retrieved.",
            )
        except Exception as exc:  # noqa: BLE001
            return self._error_response(self.SOURCE_NAME, str(exc))

    def get_historical_data(self, n_records: int = 200) -> ProviderResponse:
        retrieved_at = self._now_utc_iso()
        try:
            df   = self._load_sorted()
            rows = df.tail(n_records)
            obs_list = [self._row_to_obs(rows.iloc[i], retrieved_at)
                        for i in range(len(rows))]
            return ProviderResponse(
                status       = "ok",
                source       = self.SOURCE_NAME,
                observations = obs_list,
                message      = f"{len(obs_list)} local observations retrieved.",
            )
        except Exception as exc:  # noqa: BLE001
            return self._error_response(self.SOURCE_NAME, str(exc))

    def get_events(
        self,
        start_date: str | None = None,
        end_date:   str | None = None,
    ) -> ProviderResponse:
        # Local CSV contains no event data.
        return ProviderResponse(
            status  = "ok",
            source  = self.SOURCE_NAME,
            events  = [],
            message = "Local provider does not supply event data.",
        )
