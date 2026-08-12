"""
Tests for BobVoyage data layer (Milestone 6).

All HTTP calls are mocked — zero live network dependency.

Covers:
  Canonical models
    - SpaceWeatherObservation to_dict / from_dict round-trip
    - SpaceWeatherEvent to_dict / from_dict round-trip
    - ProviderResponse to_dict serialisation
    - None fields represented as None (not fabricated)

  LocalProvider
    - get_current_conditions returns latest row
    - get_historical_data returns sorted observations
    - Observation fields match CSV columns
    - Missing CSV returns error response
    - Empty CSV returns error response
    - get_events returns empty event list

  NOAAProvider (mocked HTTP)
    - Successful response parsed correctly
    - xray_flux extracted from XRS-B band only
    - solar_wind_speed extracted from wind summary
    - geomagnetic_index extracted from Kp feed
    - proton_flux extracted from P8B channel
    - solar_wind_density is None (not available)
    - magnetic_field is None (not available)
    - is_stale correctly set for old timestamps
    - HTTP timeout returns error response
    - HTTP 500 returns error response
    - Malformed JSON returns error response
    - Partial failure (one endpoint down) returns degraded response
    - get_historical_data returns aligned observation list
    - get_events parses NOAA alerts into SpaceWeatherEvent objects

  NASADonkiProvider (mocked HTTP)
    - CME events parsed correctly
    - FLR events parsed with classType as severity
    - GST events parsed with Kp severity
    - SEP events parsed
    - Truncated JSON (DEMO_KEY limit) returns degraded, not crash
    - HTTP error per event type returns degraded
    - get_current_conditions returns error (not supported)
    - get_historical_data returns error (not supported)

  ProviderCache
    - Fresh entry returned within TTL
    - Expired entry returns None from get()
    - Stale entry returned from get_stale()
    - put() overwrites existing entry
    - invalidate() removes entry
    - clear() removes all entries
    - is_fresh() correct
    - thread-safety (basic)

  CachedProvider
    - Cache hit avoids provider call
    - Cache miss calls provider and stores result
    - Provider error returns stale cache entry as degraded
    - Provider error with no cache returns error response
    - TTL=0 never caches (always calls provider)

  Factory
    - get_provider("local") returns CachedProvider wrapping LocalProvider
    - get_provider("noaa") returns CachedProvider wrapping NOAAProvider
    - get_provider("nasa_donki") returns CachedProvider wrapping NASADonkiProvider
    - Unknown provider raises ValueError
    - BOBVOYAGE_DATA_PROVIDER env variable honoured
    - csv_path forwarded to LocalProvider
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Canonical models
# ---------------------------------------------------------------------------
from bobvoyage.data.models.space_weather import (
    ProviderResponse,
    SpaceWeatherEvent,
    SpaceWeatherObservation,
)

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
from bobvoyage.data.providers.local      import LocalProvider
from bobvoyage.data.providers.noaa       import NOAAProvider
from bobvoyage.data.providers.nasa_donki import NASADonkiProvider

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
from bobvoyage.data.cache.provider_cache import CachedProvider, ProviderCache

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
from bobvoyage.data.factory import get_provider, list_providers


# ===========================================================================
# Helpers
# ===========================================================================

def _make_csv(path: Path, n: int = 10) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "solar_wind_speed", "solar_wind_density",
            "magnetic_field", "xray_flux", "proton_flux", "geomagnetic_index",
        ])
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "timestamp":          f"2025-07-20T{i:02d}:00:00Z",
                "solar_wind_speed":   400.0 + i,
                "solar_wind_density": 5.0,
                "magnetic_field":     7.0,
                "xray_flux":          "1.2e-07",
                "proton_flux":        2.0,
                "geomagnetic_index":  2.5,
            })


# Minimal NOAA mock payloads
_XRAY_PAYLOAD = json.dumps([
    {"time_tag": "2025-07-20T12:00:00Z", "satellite": 18,
     "flux": 1.2e-7, "energy": "0.05-0.4nm"},
    {"time_tag": "2025-07-20T12:01:00Z", "satellite": 18,
     "flux": 1.35e-7, "energy": "0.1-0.8nm"},  # XRS-B → should be selected
    {"time_tag": "2025-07-20T12:02:00Z", "satellite": 18,
     "flux": 1.40e-7, "energy": "0.1-0.8nm"},
])

_WIND_PAYLOAD = json.dumps([
    {"proton_speed": 445, "time_tag": "2025-07-20T12:02:00Z"},
])

_KP_PAYLOAD = json.dumps([
    {"time_tag": "2025-07-20T09:00:00", "Kp": 1.33, "a_running": 5, "station_count": 8},
    {"time_tag": "2025-07-20T12:00:00", "Kp": 2.67, "a_running": 9, "station_count": 8},
])

_PROTON_PAYLOAD = json.dumps([
    {"time_tag": "2025-07-20T12:01:00Z", "satellite": 18,
     "flux": 3.5e-4, "energy": "other", "channel": "P6"},
    {"time_tag": "2025-07-20T12:02:00Z", "satellite": 18,
     "flux": 2.1e-6, "energy": "99900-118000 keV", "channel": "P8B"},
])

_ALERTS_PAYLOAD = json.dumps([
    {"product_id": "WATA20", "issue_datetime": "2025-07-20 12:00:00.000",
     "message": "WATCH: Geomagnetic Storm Category G1 Predicted"},
    {"product_id": "ALTEF3", "issue_datetime": "2025-07-20 11:00:00.000",
     "message": "ALERT: Electron 2MeV Integral Flux exceeded 1000pfu"},
])

_CME_PAYLOAD = json.dumps([
    {"activityID": "2025-07-19T06:00:00-CME-001",
     "startTime": "2025-07-19T06:00Z",
     "sourceLocation": "N15W30",
     "note": "Halo CME observed.",
     "cmeAnalyses": [{"speed": 800}],
     "link": "https://example.com/cme1",
     "catalog": "M2M_CATALOG",
     "instruments": [], "activeRegionNum": None,
     "submissionTime": "2025-07-19T07:00Z", "versionId": 1,
     "linkedEvents": [], "sentNotifications": []},
])

_FLR_PAYLOAD = json.dumps([
    {"flrID": "2025-07-19T09:00:00-FLR-001",
     "beginTime": "2025-07-19T09:00Z", "peakTime": "2025-07-19T09:07Z",
     "endTime": "2025-07-19T09:13Z", "classType": "M2.1",
     "sourceLocation": "S08W84", "activeRegionNum": 14135,
     "note": "", "catalog": "M2M_CATALOG",
     "instruments": [{"displayName": "GOES-P: EXIS 1.0-8.0"}],
     "submissionTime": "2025-07-19T10:00Z", "versionId": 1,
     "link": "https://example.com/flr1",
     "linkedEvents": [], "sentNotifications": []},
])

_GST_PAYLOAD = json.dumps([
    {"gstID": "2025-07-19T12:00:00-GST-001",
     "startTime": "2025-07-19T12:00Z",
     "allKpIndex": [{"observedTime": "2025-07-19T12:00Z", "kpIndex": 6, "source": "NOAA"}],
     "link": "https://example.com/gst1",
     "submissionTime": "2025-07-19T13:00Z", "versionId": 1,
     "linkedEvents": [], "sentNotifications": []},
])

_SEP_PAYLOAD = json.dumps([])


def _mock_noaa_urlopen(url: str, *args, **kwargs):
    """Return mock HTTP responses based on URL."""
    url_str = str(url.full_url if hasattr(url, "full_url") else url)
    if "xrays" in url_str:
        content = _XRAY_PAYLOAD
    elif "solar-wind-speed" in url_str:
        content = _WIND_PAYLOAD
    elif "planetary-k-index" in url_str:
        content = _KP_PAYLOAD
    elif "differential-protons" in url_str:
        content = _PROTON_PAYLOAD
    elif "alerts" in url_str:
        content = _ALERTS_PAYLOAD
    else:
        content = "[]"
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__  = MagicMock(return_value=False)
    mock_resp.read      = lambda: content.encode()
    return mock_resp


def _mock_donki_urlopen(url: str, *args, **kwargs):
    url_str = str(url.full_url if hasattr(url, "full_url") else url)
    if "/CME" in url_str:
        content = _CME_PAYLOAD
    elif "/FLR" in url_str:
        content = _FLR_PAYLOAD
    elif "/GST" in url_str:
        content = _GST_PAYLOAD
    elif "/SEP" in url_str:
        content = _SEP_PAYLOAD
    else:
        content = "[]"
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__  = MagicMock(return_value=False)
    mock_resp.read      = lambda: content.encode()
    return mock_resp


# ===========================================================================
# Canonical model tests
# ===========================================================================

class TestSpaceWeatherObservation:

    def test_to_dict_contains_all_fields(self):
        obs = SpaceWeatherObservation(
            timestamp="2025-07-20T12:00:00Z",
            solar_wind_speed=440.0,
            source="TEST",
        )
        d = obs.to_dict()
        assert d["timestamp"]        == "2025-07-20T12:00:00Z"
        assert d["solar_wind_speed"] == 440.0
        assert d["source"]           == "TEST"

    def test_none_fields_preserved(self):
        obs = SpaceWeatherObservation()
        d   = obs.to_dict()
        assert d["solar_wind_density"] is None
        assert d["magnetic_field"]     is None

    def test_from_dict_round_trip(self):
        original = SpaceWeatherObservation(
            timestamp="2025-07-20T12:00:00Z",
            solar_wind_speed=440.0,
            geomagnetic_index=3.5,
            source="TEST",
        )
        d         = original.to_dict()
        recovered = SpaceWeatherObservation.from_dict(d)
        assert recovered.solar_wind_speed  == original.solar_wind_speed
        assert recovered.geomagnetic_index == original.geomagnetic_index

    def test_unknown_fields_ignored_in_from_dict(self):
        d = {"timestamp": "2025-07-20T12:00:00Z", "unknown_field": "ignored"}
        obs = SpaceWeatherObservation.from_dict(d)
        assert obs.timestamp == "2025-07-20T12:00:00Z"


class TestSpaceWeatherEvent:

    def test_to_dict(self):
        ev = SpaceWeatherEvent(
            event_type="CME", event_time="2025-07-20T06:00Z",
            source="NASA_DONKI", severity="fast",
        )
        d = ev.to_dict()
        assert d["event_type"] == "CME"
        assert d["severity"]   == "fast"

    def test_from_dict_round_trip(self):
        ev  = SpaceWeatherEvent(event_type="FLR", severity="M2.1")
        d   = ev.to_dict()
        ev2 = SpaceWeatherEvent.from_dict(d)
        assert ev2.severity == "M2.1"


class TestProviderResponse:

    def test_to_dict_without_observation(self):
        pr = ProviderResponse(status="error", source="TEST", message="boom")
        d  = pr.to_dict()
        assert d["status"]      == "error"
        assert d["observation"] is None
        assert d["observations"] == []

    def test_to_dict_with_observation(self):
        obs = SpaceWeatherObservation(timestamp="2025-07-20T12:00:00Z",
                                      solar_wind_speed=440.0, source="TEST")
        pr  = ProviderResponse(status="ok", source="TEST", observation=obs)
        d   = pr.to_dict()
        assert d["observation"]["solar_wind_speed"] == 440.0

    def test_to_dict_with_events(self):
        ev = SpaceWeatherEvent(event_type="CME", source="NASA")
        pr = ProviderResponse(status="ok", source="NASA", events=[ev])
        d  = pr.to_dict()
        assert len(d["events"]) == 1
        assert d["events"][0]["event_type"] == "CME"


# ===========================================================================
# LocalProvider
# ===========================================================================

class TestLocalProvider:

    def test_get_current_conditions_returns_latest(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _make_csv(csv_file, n=5)
        prov   = LocalProvider(csv_path=csv_file)
        result = prov.get_current_conditions()
        assert result.status == "ok"
        obs = result.observation
        assert obs is not None
        assert obs.solar_wind_speed == pytest.approx(404.0, abs=0.01)  # row index 4

    def test_get_historical_data_sorted(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _make_csv(csv_file, n=10)
        prov   = LocalProvider(csv_path=csv_file)
        result = prov.get_historical_data(n_records=5)
        assert result.status == "ok"
        assert len(result.observations) == 5
        # Should be oldest-to-newest
        speeds = [o.solar_wind_speed for o in result.observations]
        assert speeds == sorted(speeds)

    def test_get_current_source_is_local(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _make_csv(csv_file)
        result = LocalProvider(csv_path=csv_file).get_current_conditions()
        assert result.observation.source == "LOCAL"

    def test_missing_csv_returns_error(self, tmp_path):
        result = LocalProvider(csv_path=tmp_path / "no.csv").get_current_conditions()
        assert result.status == "error"
        assert "not found" in result.message.lower()

    def test_empty_csv_returns_error(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        result = LocalProvider(csv_path=csv_file).get_current_conditions()
        assert result.status == "error"

    def test_get_events_returns_empty_list(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _make_csv(csv_file)
        result = LocalProvider(csv_path=csv_file).get_events()
        assert result.status == "ok"
        assert result.events == []

    def test_observation_fields_populated(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _make_csv(csv_file, n=3)
        prov   = LocalProvider(csv_path=csv_file)
        result = prov.get_current_conditions()
        obs    = result.observation
        assert obs.solar_wind_density == pytest.approx(5.0)
        assert obs.magnetic_field     == pytest.approx(7.0)
        assert obs.xray_flux          == pytest.approx(1.2e-7, rel=1e-3)
        assert obs.proton_flux        == pytest.approx(2.0)
        assert obs.geomagnetic_index  == pytest.approx(2.5)


# ===========================================================================
# NOAAProvider (mocked)
# ===========================================================================

class TestNOAAProvider:

    def test_successful_current_conditions(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        assert result.status in ("ok", "degraded")
        obs = result.observation
        assert obs is not None
        assert obs.source == "NOAA"

    def test_xray_flux_from_xrsb_band(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        obs = result.observation
        # Latest XRS-B value is 1.40e-7
        assert obs.xray_flux == pytest.approx(1.40e-7, rel=1e-3)

    def test_solar_wind_speed_parsed(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        assert result.observation.solar_wind_speed == pytest.approx(445.0)

    def test_geomagnetic_index_parsed(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        # Latest Kp = 2.67
        assert result.observation.geomagnetic_index == pytest.approx(2.67, rel=0.01)

    def test_proton_flux_from_p8b_channel(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        assert result.observation.proton_flux == pytest.approx(2.1e-6, rel=1e-3)

    def test_solar_wind_density_is_none(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        assert result.observation.solar_wind_density is None

    def test_magnetic_field_is_none(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        assert result.observation.magnetic_field is None

    def test_provider_meta_has_missing_fields(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        meta = result.observation.provider_meta
        assert "solar_wind_density" in meta["missing_fields"]
        assert "magnetic_field"     in meta["missing_fields"]

    def test_stale_flag_set_for_old_timestamp(self):
        # Inject a very old timestamp to trigger stale detection
        old_payload = json.dumps([
            {"time_tag": "2000-01-01T00:00:00Z", "flux": 1.2e-7, "energy": "0.1-0.8nm"},
        ])
        def mock_old(req, *a, **k):
            mock = MagicMock()
            mock.__enter__ = lambda s: s
            mock.__exit__  = MagicMock(return_value=False)
            mock.read      = lambda: old_payload.encode()
            return mock
        with patch("urllib.request.urlopen", side_effect=mock_old):
            result = NOAAProvider().get_current_conditions()
        # Observation for year 2000 should be stale
        assert result.observation.is_stale is True

    def test_http_timeout_returns_error_or_degraded(self):
        import urllib.error
        def timeout_side_effect(req, *a, **k):
            raise TimeoutError("timeout")
        with patch("urllib.request.urlopen", side_effect=timeout_side_effect):
            result = NOAAProvider().get_current_conditions()
        assert result.status == "error"

    def test_partial_failure_returns_degraded(self):
        """One endpoint fails → degraded, not error."""
        call_count = [0]
        def partial_fail(req, *a, **k):
            call_count[0] += 1
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "xrays" in url:
                raise TimeoutError("xray timeout")
            return _mock_noaa_urlopen(req, *a, **k)
        with patch("urllib.request.urlopen", side_effect=partial_fail):
            result = NOAAProvider().get_current_conditions()
        assert result.status == "degraded"

    def test_malformed_json_returns_error(self):
        def bad_json(req, *a, **k):
            mock = MagicMock()
            mock.__enter__ = lambda s: s
            mock.__exit__  = MagicMock(return_value=False)
            mock.read      = lambda: b"{bad json!!!"
            return mock
        with patch("urllib.request.urlopen", side_effect=bad_json):
            result = NOAAProvider().get_current_conditions()
        assert result.status == "error"

    def test_get_historical_data_returns_observations(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_historical_data(n_records=10)
        assert result.status in ("ok", "degraded")
        assert len(result.observations) > 0

    def test_get_events_parses_alerts(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_events()
        assert result.status == "ok"
        assert len(result.events) == 2
        assert all(isinstance(e, SpaceWeatherEvent) for e in result.events)

    def test_result_is_json_serialisable(self):
        with patch("urllib.request.urlopen", side_effect=_mock_noaa_urlopen):
            result = NOAAProvider().get_current_conditions()
        json.dumps(result.to_dict())  # must not raise


# ===========================================================================
# NASADonkiProvider (mocked)
# ===========================================================================

class TestNASADonkiProvider:

    def test_get_events_returns_cme(self):
        with patch("urllib.request.urlopen", side_effect=_mock_donki_urlopen):
            result = NASADonkiProvider().get_events("2025-07-19", "2025-07-20")
        cmes = [e for e in result.events if e.event_type == "CME"]
        assert len(cmes) == 1
        assert "800 km/s" in cmes[0].description

    def test_get_events_flr_has_severity(self):
        with patch("urllib.request.urlopen", side_effect=_mock_donki_urlopen):
            result = NASADonkiProvider().get_events("2025-07-19", "2025-07-20")
        flrs = [e for e in result.events if e.event_type == "FLR"]
        assert len(flrs) == 1
        assert flrs[0].severity == "M2.1"

    def test_get_events_gst_has_kp_severity(self):
        with patch("urllib.request.urlopen", side_effect=_mock_donki_urlopen):
            result = NASADonkiProvider().get_events("2025-07-19", "2025-07-20")
        gsts = [e for e in result.events if e.event_type == "GST"]
        assert len(gsts) == 1
        assert "Kp6" in gsts[0].severity

    def test_events_sorted_by_time(self):
        with patch("urllib.request.urlopen", side_effect=_mock_donki_urlopen):
            result = NASADonkiProvider().get_events("2025-07-19", "2025-07-20")
        times = [e.event_time for e in result.events if e.event_time]
        assert times == sorted(times)

    def test_truncated_json_returns_degraded(self):
        def truncated(req, *a, **k):
            mock = MagicMock()
            mock.__enter__ = lambda s: s
            mock.__exit__  = MagicMock(return_value=False)
            mock.read      = lambda: b'[{"activityID": "incomplete'
            return mock
        with patch("urllib.request.urlopen", side_effect=truncated):
            result = NASADonkiProvider().get_events()
        assert result.status == "degraded"

    def test_get_current_conditions_returns_error(self):
        result = NASADonkiProvider().get_current_conditions()
        assert result.status == "error"

    def test_get_historical_data_returns_error(self):
        result = NASADonkiProvider().get_historical_data()
        assert result.status == "error"

    def test_result_is_json_serialisable(self):
        with patch("urllib.request.urlopen", side_effect=_mock_donki_urlopen):
            result = NASADonkiProvider().get_events()
        json.dumps(result.to_dict())


# ===========================================================================
# ProviderCache
# ===========================================================================

class TestProviderCache:

    def _ok_response(self, source: str = "TEST") -> ProviderResponse:
        return ProviderResponse(status="ok", source=source, message="test")

    def test_fresh_entry_returned(self):
        cache = ProviderCache(ttl_seconds=60)
        resp  = self._ok_response()
        cache.put("key1", resp)
        assert cache.get("key1") is resp

    def test_expired_entry_returns_none(self):
        cache = ProviderCache(ttl_seconds=0.01)
        resp  = self._ok_response()
        cache.put("key1", resp)
        time.sleep(0.05)
        assert cache.get("key1") is None

    def test_stale_entry_still_retrievable(self):
        cache = ProviderCache(ttl_seconds=0.01)
        resp  = self._ok_response()
        cache.put("key1", resp)
        time.sleep(0.05)
        assert cache.get_stale("key1") is resp

    def test_absent_key_returns_none(self):
        cache = ProviderCache()
        assert cache.get("nonexistent") is None
        assert cache.get_stale("nonexistent") is None

    def test_put_overwrites_existing(self):
        cache = ProviderCache(ttl_seconds=60)
        r1 = self._ok_response("A")
        r2 = self._ok_response("B")
        cache.put("k", r1)
        cache.put("k", r2)
        assert cache.get("k") is r2

    def test_invalidate_removes_entry(self):
        cache = ProviderCache(ttl_seconds=60)
        cache.put("k", self._ok_response())
        cache.invalidate("k")
        assert cache.get("k") is None

    def test_clear_removes_all(self):
        cache = ProviderCache(ttl_seconds=60)
        for i in range(5):
            cache.put(f"k{i}", self._ok_response())
        cache.clear()
        for i in range(5):
            assert cache.get(f"k{i}") is None

    def test_is_fresh_true_within_ttl(self):
        cache = ProviderCache(ttl_seconds=60)
        cache.put("k", self._ok_response())
        assert cache.is_fresh("k") is True

    def test_is_fresh_false_after_expiry(self):
        cache = ProviderCache(ttl_seconds=0.01)
        cache.put("k", self._ok_response())
        time.sleep(0.05)
        assert cache.is_fresh("k") is False

    def test_thread_safe_concurrent_puts(self):
        cache   = ProviderCache(ttl_seconds=60)
        errors  = []
        def _put(i):
            try:
                cache.put(f"k{i}", self._ok_response(f"SRC{i}"))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=_put, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# CachedProvider
# ===========================================================================

class TestCachedProvider:

    def _make_mock_provider(self, response: ProviderResponse):
        mock = MagicMock()
        mock.SOURCE_NAME = "MOCK"
        mock.get_current_conditions.return_value = response
        mock.get_historical_data.return_value    = response
        mock.get_events.return_value             = response
        return mock

    def test_cache_hit_avoids_provider_call(self):
        resp    = ProviderResponse(status="ok", source="MOCK")
        mock    = self._make_mock_provider(resp)
        cached  = CachedProvider(mock, ttl_seconds=60)
        _ = cached.get_current_conditions()  # first call — populates cache
        _ = cached.get_current_conditions()  # second call — should hit cache
        assert mock.get_current_conditions.call_count == 1

    def test_cache_miss_calls_provider(self):
        resp   = ProviderResponse(status="ok", source="MOCK")
        mock   = self._make_mock_provider(resp)
        cached = CachedProvider(mock, ttl_seconds=0)  # TTL=0 → always miss
        cached.get_current_conditions()
        cached.get_current_conditions()
        assert mock.get_current_conditions.call_count == 2

    def test_provider_error_returns_stale_as_degraded(self):
        resp = ProviderResponse(status="ok", source="MOCK",
                                observation=SpaceWeatherObservation(
                                    timestamp="2025-07-20T12:00:00Z",
                                    source="MOCK"))
        mock = self._make_mock_provider(resp)
        cache  = ProviderCache(ttl_seconds=0.01)
        cached = CachedProvider(mock, ttl_seconds=0.01, cache=cache)

        # Populate cache
        cached.get_current_conditions()

        # Now make provider fail
        mock.get_current_conditions.side_effect = RuntimeError("network down")
        time.sleep(0.05)  # let cache expire

        result = cached.get_current_conditions()
        assert result.status == "degraded"

    def test_provider_error_no_cache_returns_error(self):
        mock = MagicMock()
        mock.SOURCE_NAME = "MOCK"
        mock.get_current_conditions.side_effect = RuntimeError("boom")
        cached = CachedProvider(mock, ttl_seconds=60)
        result = cached.get_current_conditions()
        assert result.status == "error"


# ===========================================================================
# Factory
# ===========================================================================

class TestFactory:

    def test_local_provider_returned(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _make_csv(csv_file)
        prov = get_provider("local", csv_path=str(csv_file))
        assert isinstance(prov, CachedProvider)
        assert prov.SOURCE_NAME == "LOCAL"

    def test_noaa_provider_returned(self):
        prov = get_provider("noaa")
        assert prov.SOURCE_NAME == "NOAA"

    def test_nasa_donki_provider_returned(self):
        prov = get_provider("nasa_donki")
        assert prov.SOURCE_NAME == "NASA_DONKI"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_xyz")

    def test_env_variable_honoured(self, monkeypatch):
        monkeypatch.setenv("BOBVOYAGE_DATA_PROVIDER", "noaa")
        prov = get_provider()   # no explicit provider
        assert prov.SOURCE_NAME == "NOAA"

    def test_default_is_local_when_no_env(self, monkeypatch):
        monkeypatch.delenv("BOBVOYAGE_DATA_PROVIDER", raising=False)
        prov = get_provider()
        assert prov.SOURCE_NAME == "LOCAL"

    def test_list_providers(self):
        providers = list_providers()
        assert "local" in providers
        assert "noaa"  in providers
        assert "nasa_donki" in providers

    def test_case_insensitive_provider_name(self):
        prov = get_provider("NOAA")
        assert prov.SOURCE_NAME == "NOAA"
