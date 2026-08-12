"""
NOAAProvider — retrieves space-weather data from NOAA SWPC.

Confirmed working endpoints (verified 2025-07):

  X-ray flux (6 h):
    https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json
    → fields: time_tag, flux (W/m²), energy ("0.1-0.8nm" is XRS-B)

  Kp index (7-day observed):
    https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
    → fields: time_tag, Kp

  Solar wind speed (summary):
    https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json
    → fields: proton_speed (km/s), time_tag

  Differential proton flux (6 h):
    https://services.swpc.noaa.gov/json/goes/primary/differential-protons-6-hour.json
    → fields: time_tag, flux, energy, channel
    → channel "P8B" ≈ 99.9–118 MeV  (closest to standard integral pfu)

  Alerts:
    https://services.swpc.noaa.gov/products/alerts.json
    → fields: product_id, issue_datetime, message

LIMITATION:
  Solar wind density and Bz (magnetic field) are NOT available from NOAA
  SWPC as clean JSON in 2025.  Those fields will be set to None and the
  provider_meta will note which source would supply them when available.

No authentication required.
No rate-limit documented; standard practice is ≥ 1 s between requests.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from bobvoyage.data.models.space_weather import (
    ProviderResponse,
    SpaceWeatherEvent,
    SpaceWeatherObservation,
)
from bobvoyage.data.providers.base import SpaceWeatherProvider

# ---------------------------------------------------------------------------
# Endpoint constants
# ---------------------------------------------------------------------------
_BASE = "https://services.swpc.noaa.gov"

ENDPOINTS = {
    "xray_6h":         f"{_BASE}/json/goes/primary/xrays-6-hour.json",
    "kp_7day":         f"{_BASE}/products/noaa-planetary-k-index.json",
    "wind_speed":      f"{_BASE}/products/summary/solar-wind-speed.json",
    "proton_6h":       f"{_BASE}/json/goes/primary/differential-protons-6-hour.json",
    "alerts":          f"{_BASE}/products/alerts.json",
}

# GOES differential proton channel that maps closest to integral pfu (~100 MeV)
_PROTON_CHANNEL = "P8B"

# Freshness threshold: NOAA updates every ~1 min for GOES data
_STALE_SECONDS = 600  # 10 minutes


class NOAAProvider(SpaceWeatherProvider):
    """Retrieves real-time space-weather data from NOAA SWPC JSON feeds."""

    SOURCE_NAME              = "NOAA"
    STALE_THRESHOLD_SECONDS  = _STALE_SECONDS

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _fetch_json(self, url: str) -> list | dict:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BobVoyage/0.1 (github.com/YahirRR/BobVoyage)"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_xray_current(self, data: list[dict]) -> tuple[float | None, str | None]:
        """Return (flux_W_m2, timestamp) for the latest XRS-B reading."""
        xrsb = [r for r in data if r.get("energy") == "0.1-0.8nm"]
        if not xrsb:
            return None, None
        latest = xrsb[-1]
        return self._safe_float(latest.get("flux")), latest.get("time_tag")

    def _parse_xray_historical(self, data: list[dict]) -> list[tuple[str | None, float | None]]:
        """Return [(timestamp, flux)] for XRS-B records, oldest first."""
        xrsb = [r for r in data if r.get("energy") == "0.1-0.8nm"]
        return [(r.get("time_tag"), self._safe_float(r.get("flux"))) for r in xrsb]

    def _parse_kp_current(self, data: list[dict]) -> tuple[float | None, str | None]:
        observed = [r for r in data if isinstance(r, dict) and r.get("Kp") is not None]
        if not observed:
            return None, None
        latest = observed[-1]
        return self._safe_float(latest.get("Kp")), latest.get("time_tag")

    def _parse_kp_historical(self, data: list[dict]) -> dict[str, float]:
        """Return {time_tag: Kp} for all observed records."""
        return {
            r["time_tag"]: self._safe_float(r.get("Kp"))
            for r in data
            if isinstance(r, dict) and r.get("time_tag") and r.get("Kp") is not None
        }

    def _parse_wind_speed(self, data: list | dict) -> tuple[float | None, str | None]:
        if isinstance(data, list) and data:
            rec = data[-1]
        elif isinstance(data, dict):
            rec = data
        else:
            return None, None
        return self._safe_float(rec.get("proton_speed")), rec.get("time_tag")

    def _parse_proton_current(self, data: list[dict]) -> tuple[float | None, str | None]:
        """Return (flux, timestamp) for the latest P8B differential proton reading."""
        p8b = [r for r in data if r.get("channel") == _PROTON_CHANNEL]
        if not p8b:
            return None, None
        latest = p8b[-1]
        return self._safe_float(latest.get("flux")), latest.get("time_tag")

    def _parse_proton_historical(self, data: list[dict]) -> dict[str, float]:
        return {
            r["time_tag"]: self._safe_float(r.get("flux"))
            for r in data
            if r.get("channel") == _PROTON_CHANNEL and r.get("time_tag")
        }

    # ------------------------------------------------------------------
    # Provider interface — get_current_conditions
    # ------------------------------------------------------------------

    def get_current_conditions(self) -> ProviderResponse:
        retrieved_at = self._now_utc_iso()
        errors: list[str] = []

        xray_flux        = None
        xray_ts          = None
        wind_speed       = None
        wind_ts          = None
        kp               = None
        kp_ts            = None
        proton           = None
        proton_ts        = None

        # --- X-ray flux -------------------------------------------------------
        try:
            xray_data        = self._fetch_json(ENDPOINTS["xray_6h"])
            xray_flux, xray_ts = self._parse_xray_current(xray_data)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"xray_flux: {exc}")

        # --- Solar wind speed -------------------------------------------------
        try:
            wind_data        = self._fetch_json(ENDPOINTS["wind_speed"])
            wind_speed, wind_ts = self._parse_wind_speed(wind_data)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"solar_wind_speed: {exc}")

        # --- Kp index ---------------------------------------------------------
        try:
            kp_data          = self._fetch_json(ENDPOINTS["kp_7day"])
            kp, kp_ts        = self._parse_kp_current(kp_data)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"geomagnetic_index: {exc}")

        # --- Proton flux ------------------------------------------------------
        try:
            proton_data      = self._fetch_json(ENDPOINTS["proton_6h"])
            proton, proton_ts = self._parse_proton_current(proton_data)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"proton_flux: {exc}")

        # --- Best available timestamp -----------------------------------------
        # Use X-ray timestamp as primary (highest resolution: 1-min)
        best_ts = xray_ts or wind_ts or kp_ts or proton_ts

        # All data failed
        if best_ts is None and all(
            v is None for v in [xray_flux, wind_speed, kp, proton]
        ):
            return self._error_response(
                self.SOURCE_NAME,
                f"All NOAA endpoints failed: {'; '.join(errors)}",
            )

        obs = SpaceWeatherObservation(
            timestamp           = best_ts,
            solar_wind_speed    = wind_speed,
            solar_wind_density  = None,   # not available from NOAA SWPC JSON
            magnetic_field      = None,   # not available from NOAA SWPC JSON
            xray_flux           = xray_flux,
            proton_flux         = proton,
            geomagnetic_index   = kp,
            source              = self.SOURCE_NAME,
            retrieved_at        = retrieved_at,
            provider_meta       = {
                "xray_timestamp":   xray_ts,
                "wind_timestamp":   wind_ts,
                "kp_timestamp":     kp_ts,
                "proton_timestamp": proton_ts,
                "missing_fields":   ["solar_wind_density", "magnetic_field"],
                "missing_reason":   (
                    "NOAA SWPC does not provide solar wind density "
                    "or Bz as clean JSON in the current API (2025)."
                ),
            },
        )
        obs = self._mark_staleness(obs)

        status  = "degraded" if errors else "ok"
        message = (
            "Current conditions retrieved from NOAA SWPC."
            if not errors
            else f"Partial data — errors: {'; '.join(errors)}"
        )

        return ProviderResponse(
            status      = status,
            source      = self.SOURCE_NAME,
            observation = obs,
            message     = message,
        )

    # ------------------------------------------------------------------
    # get_historical_data
    # ------------------------------------------------------------------

    def get_historical_data(self, n_records: int = 200) -> ProviderResponse:
        retrieved_at = self._now_utc_iso()
        errors: list[str] = []

        xray_history:   list[tuple]       = []
        kp_lookup:      dict[str, float]  = {}
        proton_lookup:  dict[str, float]  = {}

        try:
            xray_history = self._parse_xray_historical(
                self._fetch_json(ENDPOINTS["xray_6h"])
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"xray: {exc}")

        try:
            kp_lookup = self._parse_kp_historical(
                self._fetch_json(ENDPOINTS["kp_7day"])
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"kp: {exc}")

        try:
            proton_lookup = self._parse_proton_historical(
                self._fetch_json(ENDPOINTS["proton_6h"])
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"proton: {exc}")

        if not xray_history:
            return self._error_response(
                self.SOURCE_NAME,
                f"No X-ray history available: {'; '.join(errors)}",
            )

        # Build aligned observations.  Primary timeline = X-ray (1-min).
        # Kp (3-hour) and proton (1-min) are joined by nearest timestamp match
        # (simplified: exact match; Kp will be sparse but not fabricated).
        obs_list: list[SpaceWeatherObservation] = []
        rows = xray_history[-n_records:]   # oldest first, capped to n_records

        # Fetch current wind speed (summary has only one point)
        try:
            ws_data = self._fetch_json(ENDPOINTS["wind_speed"])
            current_wind, _ = self._parse_wind_speed(ws_data)
        except Exception:  # noqa: BLE001
            current_wind = None

        for ts, flux in rows:
            kp_val     = kp_lookup.get(ts)           # exact match or None
            proton_val = proton_lookup.get(ts)        # exact match or None

            obs_list.append(SpaceWeatherObservation(
                timestamp           = ts,
                solar_wind_speed    = current_wind,   # only latest available
                solar_wind_density  = None,
                magnetic_field      = None,
                xray_flux           = flux,
                proton_flux         = proton_val,
                geomagnetic_index   = kp_val,
                source              = self.SOURCE_NAME,
                retrieved_at        = retrieved_at,
                data_age_seconds    = self._age_seconds(ts),
                is_stale            = False,
            ))

        status  = "degraded" if errors else "ok"
        return ProviderResponse(
            status       = status,
            source       = self.SOURCE_NAME,
            observations = obs_list,
            message      = (
                f"{len(obs_list)} observations from NOAA SWPC."
                + (f" Partial errors: {'; '.join(errors)}" if errors else "")
            ),
        )

    # ------------------------------------------------------------------
    # get_events — NOAA alerts
    # ------------------------------------------------------------------

    def get_events(
        self,
        start_date: str | None = None,
        end_date:   str | None = None,
    ) -> ProviderResponse:
        retrieved_at = self._now_utc_iso()
        try:
            raw_alerts: list[dict] = self._fetch_json(ENDPOINTS["alerts"])
        except Exception as exc:  # noqa: BLE001
            return self._error_response(
                self.SOURCE_NAME, f"Failed to fetch NOAA alerts: {exc}"
            )

        events: list[SpaceWeatherEvent] = []
        for alert in raw_alerts:
            msg      = alert.get("message", "")
            pid      = alert.get("product_id", "")
            issued   = alert.get("issue_datetime", "")

            # Normalise product_id → event_type
            pid_upper = pid.upper()
            if "CME" in pid_upper or "ALT" in pid_upper:
                etype = "ALERT"
            elif "WAT" in pid_upper or "WAR" in pid_upper:
                etype = "ALERT"
            elif "SUM" in pid_upper:
                etype = "ALERT"
            else:
                etype = "ALERT"

            events.append(SpaceWeatherEvent(
                event_type  = etype,
                event_time  = issued,
                source      = self.SOURCE_NAME,
                external_id = pid,
                description = msg[:500],   # truncate long messages
                extra       = {"product_id": pid},
            ))

        return ProviderResponse(
            status  = "ok",
            source  = self.SOURCE_NAME,
            events  = events,
            message = f"{len(events)} NOAA alerts retrieved.",
        )
