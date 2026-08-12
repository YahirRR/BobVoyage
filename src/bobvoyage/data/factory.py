"""
BobVoyage provider factory.

Selects and configures the appropriate data provider based on:

  1. The BOBVOYAGE_DATA_PROVIDER environment variable.
  2. An explicit `provider` argument.
  3. Default: "local" (deterministic, no network dependency).

Supported values for BOBVOYAGE_DATA_PROVIDER:
  local        LocalProvider (CSV — always deterministic)
  noaa         NOAAProvider (live NOAA SWPC feeds, with cache)
  nasa_donki   NASADonkiProvider (NASA event feed, with cache)

The returned object is always wrapped in a CachedProvider, so callers
never need to worry about cache management.
"""

from __future__ import annotations

import os

from bobvoyage.data.cache.provider_cache import CachedProvider, ProviderCache
from bobvoyage.data.providers.base      import SpaceWeatherProvider
from bobvoyage.data.providers.local     import LocalProvider
from bobvoyage.data.providers.noaa      import NOAAProvider
from bobvoyage.data.providers.nasa_donki import NASADonkiProvider

# Default TTL values per provider type (seconds)
_TTL = {
    "local":      float("inf"),  # local data never expires
    "noaa":       60.0,          # NOAA updates every ~1 min
    "nasa_donki": 300.0,         # DONKI events change infrequently
}

_ENV_KEY = "BOBVOYAGE_DATA_PROVIDER"


def get_provider(
    provider:     str | None = None,
    csv_path:     str | None = None,
    nasa_api_key: str | None = None,
    timeout:      float = 10.0,
    ttl_seconds:  float | None = None,
    cache:        ProviderCache | None = None,
) -> CachedProvider:
    """
    Return a CachedProvider wrapping the selected SpaceWeatherProvider.

    Parameters
    ----------
    provider:
        One of "local", "noaa", "nasa_donki".
        Defaults to the BOBVOYAGE_DATA_PROVIDER env variable, or "local".
    csv_path:
        Path to the local CSV file (only used when provider="local").
    nasa_api_key:
        NASA API key (only used when provider="nasa_donki").
        Falls back to BOBVOYAGE_NASA_API_KEY env var, then DEMO_KEY.
    timeout:
        HTTP request timeout in seconds (NOAA / NASA providers).
    ttl_seconds:
        Cache TTL override.  If None, uses the provider-specific default.
    cache:
        External ProviderCache instance (useful for testing).

    Returns
    -------
    CachedProvider
    """
    name = (provider or os.environ.get(_ENV_KEY, "local")).lower().strip()

    if name == "local":
        raw: SpaceWeatherProvider = LocalProvider(csv_path=csv_path)
        ttl = ttl_seconds if ttl_seconds is not None else _TTL["local"]

    elif name == "noaa":
        raw = NOAAProvider(timeout=timeout)
        ttl = ttl_seconds if ttl_seconds is not None else _TTL["noaa"]

    elif name == "nasa_donki":
        raw = NASADonkiProvider(api_key=nasa_api_key, timeout=timeout)
        ttl = ttl_seconds if ttl_seconds is not None else _TTL["nasa_donki"]

    else:
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Supported: local, noaa, nasa_donki. "
            f"Set BOBVOYAGE_DATA_PROVIDER or pass provider= explicitly."
        )

    return CachedProvider(raw, ttl_seconds=ttl, cache=cache)


def list_providers() -> list[str]:
    """Return the names of all registered providers."""
    return ["local", "noaa", "nasa_donki"]
