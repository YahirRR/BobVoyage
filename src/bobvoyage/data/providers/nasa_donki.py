"""
NASADonkiProvider — retrieves space-weather events from NASA CCMC/DONKI.

Confirmed working endpoints (verified 2025-07):

  CME (Coronal Mass Ejection):
    https://api.nasa.gov/DONKI/CME
    query params: startDate, endDate, api_key

  FLR (Solar Flare):
    https://api.nasa.gov/DONKI/FLR
    query params: startDate, endDate, api_key

  GST (Geomagnetic Storm):
    https://api.nasa.gov/DONKI/GST
    query params: startDate, endDate, api_key

  SEP (Solar Energetic Particle):
    https://api.nasa.gov/DONKI/SEP
    query params: startDate, endDate, api_key

Authentication:
  api_key=DEMO_KEY allows ~30 req/hour.
  A free registered key at https://api.nasa.gov/ allows ~1000 req/hour.
  Key is read from BOBVOYAGE_NASA_API_KEY env variable; falls back to DEMO_KEY.

Date format: YYYY-MM-DD
Default window: last 7 days.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

from bobvoyage.data.models.space_weather import (
    ProviderResponse,
    SpaceWeatherEvent,
    SpaceWeatherObservation,
)
from bobvoyage.data.providers.base import SpaceWeatherProvider

_DONKI_BASE = "https://api.nasa.gov/DONKI"

_EVENT_ENDPOINTS = {
    "CME": f"{_DONKI_BASE}/CME",
    "FLR": f"{_DONKI_BASE}/FLR",
    "GST": f"{_DONKI_BASE}/GST",
    "SEP": f"{_DONKI_BASE}/SEP",
}

_DEFAULT_WINDOW_DAYS = 7


class NASADonkiProvider(SpaceWeatherProvider):
    """Retrieves space-weather events from NASA CCMC DONKI."""

    SOURCE_NAME = "NASA_DONKI"

    def __init__(
        self,
        api_key:  str | None = None,
        timeout:  float = 12.0,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("BOBVOYAGE_NASA_API_KEY", "DEMO_KEY")
        )
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
            raw = resp.read()
        return json.loads(raw)

    def _build_url(self, base: str, start: str, end: str) -> str:
        params = urllib.parse.urlencode({
            "startDate": start,
            "endDate":   end,
            "api_key":   self._api_key,
        })
        return f"{base}?{params}"

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_dates() -> tuple[str, str]:
        today    = datetime.now(timezone.utc).date()
        week_ago = today - timedelta(days=_DEFAULT_WINDOW_DAYS)
        return str(week_ago), str(today)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_cme(self, records: list[dict]) -> list[SpaceWeatherEvent]:
        events = []
        for r in records:
            # Best speed estimate from first analysis
            analyses = r.get("cmeAnalyses") or []
            speed_str = ""
            if analyses:
                spd = analyses[0].get("speed")
                if spd:
                    speed_str = f" speed ~{spd} km/s"
            events.append(SpaceWeatherEvent(
                event_type  = "CME",
                event_time  = r.get("startTime"),
                source      = self.SOURCE_NAME,
                external_id = r.get("activityID"),
                description = (
                    f"CME detected{speed_str}. "
                    f"Source: {r.get('sourceLocation') or 'unknown'}. "
                    + (r.get("note") or "")[:200]
                ).strip(),
                extra       = {"link": r.get("link")},
            ))
        return events

    def _parse_flr(self, records: list[dict]) -> list[SpaceWeatherEvent]:
        events = []
        for r in records:
            cls_type = r.get("classType", "")
            events.append(SpaceWeatherEvent(
                event_type  = "FLR",
                event_time  = r.get("peakTime") or r.get("beginTime"),
                source      = self.SOURCE_NAME,
                external_id = r.get("flrID"),
                description = (
                    f"Solar flare class {cls_type}. "
                    f"Peak: {r.get('peakTime')}. "
                    f"Location: {r.get('sourceLocation') or 'unknown'}."
                ),
                severity    = cls_type,
                extra       = {"link": r.get("link")},
            ))
        return events

    def _parse_gst(self, records: list[dict]) -> list[SpaceWeatherEvent]:
        events = []
        for r in records:
            kp_max = None
            for activity in (r.get("allKpIndex") or []):
                kp = self._safe_float(activity.get("kpIndex"))
                if kp is not None and (kp_max is None or kp > kp_max):
                    kp_max = kp
            events.append(SpaceWeatherEvent(
                event_type  = "GST",
                event_time  = r.get("startTime"),
                source      = self.SOURCE_NAME,
                external_id = r.get("gstID"),
                description = (
                    f"Geomagnetic storm. "
                    + (f"Max Kp: {kp_max}." if kp_max else "")
                ),
                severity    = f"Kp{kp_max:.0f}" if kp_max else None,
                extra       = {"link": r.get("link")},
            ))
        return events

    def _parse_sep(self, records: list[dict]) -> list[SpaceWeatherEvent]:
        events = []
        for r in records:
            events.append(SpaceWeatherEvent(
                event_type  = "SEP",
                event_time  = r.get("eventTime"),
                source      = self.SOURCE_NAME,
                external_id = r.get("sepID"),
                description = (
                    f"Solar energetic particle event. "
                    + (r.get("instruments") and
                       f"Instruments: {[i.get('displayName') for i in r['instruments'][:2]]}" or "")
                ),
                extra       = {"link": r.get("link")},
            ))
        return events

    # ------------------------------------------------------------------
    # Provider interface — get_events (primary capability)
    # ------------------------------------------------------------------

    def get_events(
        self,
        start_date: str | None = None,
        end_date:   str | None = None,
    ) -> ProviderResponse:
        default_start, default_end = self._default_dates()
        start = start_date or default_start
        end   = end_date   or default_end

        all_events: list[SpaceWeatherEvent] = []
        errors: list[str] = []

        parsers = {
            "CME": self._parse_cme,
            "FLR": self._parse_flr,
            "GST": self._parse_gst,
            "SEP": self._parse_sep,
        }

        for etype, base_url in _EVENT_ENDPOINTS.items():
            url = self._build_url(base_url, start, end)
            try:
                data = self._fetch_json(url)
                if isinstance(data, list):
                    all_events.extend(parsers[etype](data))
                # else: some endpoints return dicts on error
            except json.JSONDecodeError as exc:
                # DEMO_KEY sometimes truncates large responses
                errors.append(f"{etype}: truncated JSON response (use a registered API key)")
            except urllib.error.HTTPError as exc:
                errors.append(f"{etype}: HTTP {exc.code}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{etype}: {exc}")

        # Sort by event_time ascending
        all_events.sort(key=lambda e: e.event_time or "")

        status  = "degraded" if errors else "ok"
        message = (
            f"{len(all_events)} events retrieved from NASA DONKI "
            f"({start} → {end})."
            + (f" Errors: {'; '.join(errors)}" if errors else "")
        )

        return ProviderResponse(
            status  = status,
            source  = self.SOURCE_NAME,
            events  = all_events,
            message = message,
        )

    # ------------------------------------------------------------------
    # Unsupported operations (NASA DONKI is event-only)
    # ------------------------------------------------------------------

    def get_current_conditions(self) -> ProviderResponse:
        return self._error_response(
            self.SOURCE_NAME,
            "NASADonkiProvider does not supply numerical observations. "
            "Use get_events() for event data.",
        )

    def get_historical_data(self, n_records: int = 200) -> ProviderResponse:
        return self._error_response(
            self.SOURCE_NAME,
            "NASADonkiProvider does not supply time-series observations. "
            "Use get_events() for event data.",
        )
