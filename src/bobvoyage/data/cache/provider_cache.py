"""
BobVoyage in-memory TTL cache for provider responses.

Keeps the latest ProviderResponse per cache key for up to `ttl_seconds`.
Thread-safe via a simple lock.

Design decisions:
  - Lightweight: stdlib only, no Redis, no file-system state.
  - Single-process: sufficient for a stdio MCP server.
  - TTL per entry: each cached item carries its own expiry time.
  - Stale-while-revalidate: returns the stale entry AND marks it degraded
    rather than returning None — mirrors the NOAA degraded-response pattern.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from bobvoyage.data.models.space_weather import ProviderResponse


@dataclass
class _CacheEntry:
    response:   ProviderResponse
    expires_at: float   # time.monotonic() expiry


class ProviderCache:
    """
    Thread-safe in-memory TTL cache for ProviderResponse objects.

    Parameters
    ----------
    ttl_seconds:
        Default time-to-live for cached entries.
        Can be overridden per put() call.
    """

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self._ttl      = ttl_seconds
        self._store:   dict[str, _CacheEntry] = {}
        self._lock     = threading.Lock()

    def get(self, key: str) -> ProviderResponse | None:
        """
        Return a cached ProviderResponse if it is still fresh.

        Returns
        -------
        ProviderResponse   fresh entry
        None               entry absent or expired (caller must revalidate)
        """
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() < entry.expires_at:
            return entry.response
        return None

    def get_stale(self, key: str) -> ProviderResponse | None:
        """
        Return a cached entry even if it has expired (stale-while-revalidate).

        The caller is responsible for marking the response as degraded.
        Returns None only if the key was never populated.
        """
        with self._lock:
            entry = self._store.get(key)
        return entry.response if entry else None

    def put(
        self,
        key:         str,
        response:    ProviderResponse,
        ttl_seconds: float | None = None,
    ) -> None:
        """Store a response under `key` with the given TTL."""
        ttl    = ttl_seconds if ttl_seconds is not None else self._ttl
        expiry = time.monotonic() + ttl
        with self._lock:
            self._store[key] = _CacheEntry(response=response, expires_at=expiry)

    def invalidate(self, key: str) -> None:
        """Remove a specific entry from the cache."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush all entries."""
        with self._lock:
            self._store.clear()

    def is_fresh(self, key: str) -> bool:
        """True if the entry exists and has not expired."""
        with self._lock:
            entry = self._store.get(key)
        return entry is not None and time.monotonic() < entry.expires_at

    def age_seconds(self, key: str) -> float | None:
        """Return how many seconds ago the entry was last populated, or None."""
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return None
        return max(0.0, (entry.expires_at - time.monotonic()))


# ---------------------------------------------------------------------------
# Module-level singleton used by CachedProvider
# ---------------------------------------------------------------------------
_default_cache = ProviderCache(ttl_seconds=60.0)


class CachedProvider:
    """
    Wraps any SpaceWeatherProvider with a transparent TTL cache.

    On a cache hit the provider is not called at all.
    On a cache miss the provider is called and the result is stored.
    On a provider error the stale cache entry (if any) is returned as
    a "degraded" response rather than propagating the error.

    Parameters
    ----------
    provider    Any SpaceWeatherProvider implementation.
    ttl_seconds Cache lifetime in seconds.
    cache       Optional external ProviderCache (e.g. for testing).
    """

    def __init__(
        self,
        provider,
        ttl_seconds: float = 60.0,
        cache: ProviderCache | None = None,
    ) -> None:
        self._provider   = provider
        self._ttl        = ttl_seconds
        self._cache      = cache or ProviderCache(ttl_seconds=ttl_seconds)
        self.SOURCE_NAME = getattr(provider, "SOURCE_NAME", "UNKNOWN")

    def _cache_key(self, method: str) -> str:
        return f"{self.SOURCE_NAME}:{method}"

    def _call_with_fallback(
        self,
        method:    str,
        live_call,
    ) -> ProviderResponse:
        key = self._cache_key(method)

        # 1. Fresh cache hit
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # 2. Cache miss — call the live provider
        try:
            result = live_call()
            if result.status in ("ok", "degraded"):
                self._cache.put(key, result, ttl_seconds=self._ttl)
            return result
        except Exception as exc:  # noqa: BLE001
            # 3. Live call failed — serve stale if available
            stale = self._cache.get_stale(key)
            if stale is not None:
                age = self._cache.age_seconds(key)
                return ProviderResponse(
                    status  = "degraded",
                    source  = self.SOURCE_NAME,
                    observation   = stale.observation,
                    observations  = stale.observations,
                    events        = stale.events,
                    data_age_seconds = age,
                    message = (
                        f"Live provider temporarily unavailable ({exc}); "
                        f"latest cached data returned."
                    ),
                )
            # 4. No cache at all
            return ProviderResponse(
                status  = "error",
                source  = self.SOURCE_NAME,
                message = f"Provider failed and no cached data available: {exc}",
            )

    def get_current_conditions(self) -> ProviderResponse:
        return self._call_with_fallback(
            "current", self._provider.get_current_conditions
        )

    def get_historical_data(self, n_records: int = 200) -> ProviderResponse:
        return self._call_with_fallback(
            f"historical_{n_records}",
            lambda: self._provider.get_historical_data(n_records),
        )

    def get_events(
        self,
        start_date: str | None = None,
        end_date:   str | None = None,
    ) -> ProviderResponse:
        key_suffix = f"{start_date}_{end_date}"
        return self._call_with_fallback(
            f"events_{key_suffix}",
            lambda: self._provider.get_events(start_date, end_date),
        )
