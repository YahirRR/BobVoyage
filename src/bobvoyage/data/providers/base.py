"""
BobVoyage provider base class.

All concrete providers inherit from SpaceWeatherProvider and implement
the three abstract methods.  No provider-specific logic lives here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from bobvoyage.data.models.space_weather import (
    ProviderResponse,
    SpaceWeatherObservation,
    SpaceWeatherEvent,
)


class SpaceWeatherProvider(ABC):
    """Abstract base for all BobVoyage data providers."""

    # Subclasses should set this to a human-readable name.
    SOURCE_NAME: str = "UNKNOWN"

    # Number of seconds after which a cached observation is considered stale.
    STALE_THRESHOLD_SECONDS: float = 300.0  # 5 minutes default

    # ---------------------------------------------------------------------------
    # Abstract interface — every provider must implement these
    # ---------------------------------------------------------------------------

    @abstractmethod
    def get_current_conditions(self) -> ProviderResponse:
        """Return the most recent available space-weather observation."""
        ...

    @abstractmethod
    def get_historical_data(self, n_records: int = 200) -> ProviderResponse:
        """Return up to n_records recent observations, oldest first."""
        ...

    @abstractmethod
    def get_events(
        self,
        start_date: str | None = None,
        end_date:   str | None = None,
    ) -> ProviderResponse:
        """Return space-weather events in the given UTC date range (ISO-8601)."""
        ...

    # ---------------------------------------------------------------------------
    # Shared helpers — available to all subclasses
    # ---------------------------------------------------------------------------

    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _age_seconds(timestamp_str: str | None) -> float | None:
        """Return seconds elapsed since `timestamp_str` (UTC ISO-8601)."""
        if not timestamp_str:
            return None
        try:
            # Handle timestamps without timezone by assuming UTC
            ts = timestamp_str.rstrip("Z")
            if "+" not in ts and "T" in ts:
                ts_dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            else:
                ts_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - ts_dt
            return max(0.0, delta.total_seconds())
        except (ValueError, TypeError):
            return None

    def _mark_staleness(
        self,
        obs: SpaceWeatherObservation,
        threshold: float | None = None,
    ) -> SpaceWeatherObservation:
        """Compute data_age_seconds and set is_stale flag."""
        threshold = threshold or self.STALE_THRESHOLD_SECONDS
        age = self._age_seconds(obs.timestamp)
        obs.data_age_seconds = age
        obs.is_stale = (age is not None and age > threshold)
        return obs

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert to float; return None on failure or NaN."""
        import math
        if value is None:
            return None
        try:
            f = float(value)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _error_response(source: str, message: str) -> ProviderResponse:
        return ProviderResponse(
            status="error",
            source=source,
            message=message,
        )

    @staticmethod
    def _degraded_response(
        source: str,
        message: str,
        observation: SpaceWeatherObservation | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            status="degraded",
            source=source,
            observation=observation,
            message=message,
        )
