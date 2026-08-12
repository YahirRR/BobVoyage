"""
BobVoyage canonical data models.

These are the normalised internal representations that all providers
must produce and all analytical tools consume.

No provider-specific types (NOAA JSON keys, NASA field names, etc.)
should appear outside of the provider module that creates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# SpaceWeatherObservation
# ---------------------------------------------------------------------------

@dataclass
class SpaceWeatherObservation:
    """
    Normalised snapshot of space-weather measurements at a single timestamp.

    All numeric fields are Optional[float].  A value of None means the
    provider did not supply that measurement — it must never be fabricated.

    Fields
    ------
    timestamp           ISO-8601 UTC string of the observation time.
    solar_wind_speed    Solar wind proton speed in km/s.
    solar_wind_density  Solar wind proton number density in cm⁻³.
    magnetic_field      Interplanetary magnetic field total magnitude in nT.
    xray_flux           GOES XRS-B (0.1–0.8 nm) flux in W/m².
    proton_flux         GOES energetic proton flux in pfu (p/cm²/s/sr)
                        measured in the ~100 MeV channel.
    geomagnetic_index   NOAA planetary Kp index (0–9 scale).
    source              Provider identifier string (e.g. "NOAA", "LOCAL").
    retrieved_at        ISO-8601 UTC string of when this record was fetched.
    data_age_seconds    Age of the measurement at retrieval time (seconds).
    is_stale            True if data_age_seconds exceeds the provider's
                        freshness threshold.
    """
    timestamp:           str | None = None
    solar_wind_speed:    float | None = None
    solar_wind_density:  float | None = None
    magnetic_field:      float | None = None
    xray_flux:           float | None = None
    proton_flux:         float | None = None
    geomagnetic_index:   float | None = None
    source:              str  = "UNKNOWN"
    retrieved_at:        str | None = None
    data_age_seconds:    float | None = None
    is_stale:            bool = False
    provider_meta:       dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpaceWeatherObservation":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# SpaceWeatherEvent
# ---------------------------------------------------------------------------

@dataclass
class SpaceWeatherEvent:
    """
    Normalised representation of a discrete space-weather event.

    Used for NASA DONKI events (CME, solar flares, geomagnetic storms, SEPs).

    Fields
    ------
    event_type      One of: CME | FLR | GST | SEP | ALERT | OTHER.
    event_time      ISO-8601 UTC string of the event start time.
    source          Provider identifier (e.g. "NASA_DONKI", "NOAA_ALERTS").
    external_id     Provider-assigned unique identifier for the event.
    description     Human-readable summary.
    severity        Optional severity label (e.g. "C3.3", "G2", "S1").
    extra           Dict for provider-specific fields not in the schema.
    """
    event_type:   str  = "OTHER"
    event_time:   str | None = None
    source:       str  = "UNKNOWN"
    external_id:  str | None = None
    description:  str  = ""
    severity:     str | None = None
    extra:        dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpaceWeatherEvent":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# ProviderResponse
# ---------------------------------------------------------------------------

@dataclass
class ProviderResponse:
    """
    Envelope returned by every provider method.

    status      "ok" | "degraded" | "error"
    source      Provider name.
    observation The canonical observation (or None on error).
    observations List of observations for historical requests.
    events       List of SpaceWeatherEvent objects.
    data_age_seconds  Age of the oldest record in observations.
    message     Human-readable status explanation.
    """
    status:            str  = "ok"
    source:            str  = "UNKNOWN"
    observation:       SpaceWeatherObservation | None = None
    observations:      list[SpaceWeatherObservation] = field(default_factory=list)
    events:            list[SpaceWeatherEvent]        = field(default_factory=list)
    data_age_seconds:  float | None = None
    message:           str  = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON output."""
        return {
            "status":           self.status,
            "source":           self.source,
            "observation":      self.observation.to_dict() if self.observation else None,
            "observations":     [o.to_dict() for o in self.observations],
            "events":           [e.to_dict() for e in self.events],
            "data_age_seconds": self.data_age_seconds,
            "message":          self.message,
        }
