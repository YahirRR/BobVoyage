"""
Tests for get_current_conditions — BobVoyage MCP tool

Covers:
  - successful retrieval from the real dev dataset
  - correct "latest row" selection when timestamps are out of order
  - graceful handling of a missing dataset file
  - graceful handling of an empty CSV
  - graceful handling of a CSV missing optional columns
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import pytest

from bobvoyage.tools.current_conditions import get_current_conditions


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetCurrentConditionsRealDataset:
    """Tests that use the actual dev dataset shipped with the project."""

    def test_returns_ok_status(self):
        result = get_current_conditions()
        assert result["status"] == "ok", result["message"]

    def test_observation_is_dict(self):
        result = get_current_conditions()
        assert isinstance(result["observation"], dict)

    def test_timestamp_present_and_non_null(self):
        result = get_current_conditions()
        obs = result["observation"]
        assert obs["timestamp"] is not None

    def test_numeric_fields_present(self):
        result = get_current_conditions()
        obs = result["observation"]
        for field in (
            "solar_wind_speed",
            "solar_wind_density",
            "magnetic_field",
            "xray_flux",
            "proton_flux",
            "geomagnetic_index",
        ):
            assert field in obs, f"Field '{field}' missing from observation"

    def test_solar_wind_speed_in_realistic_range(self):
        result = get_current_conditions()
        speed = result["observation"]["solar_wind_speed"]
        assert speed is not None
        assert 200 <= float(speed) <= 900, f"Unexpected solar wind speed: {speed}"

    def test_no_missing_fields(self):
        result = get_current_conditions()
        assert result["missing_fields"] == [], (
            f"Missing fields: {result['missing_fields']}"
        )


class TestGetCurrentConditionsLatestRow:
    """Ensures the tool returns the row with the most recent timestamp."""

    def test_returns_latest_row(self, tmp_path):
        csv_file = tmp_path / "sw.csv"
        _write_csv(csv_file, [
            {
                "timestamp": "2025-07-20T10:00:00Z",
                "solar_wind_speed": 350.0,
                "solar_wind_density": 5.0,
                "magnetic_field": 6.0,
                "xray_flux": "1.0e-7",
                "proton_flux": 1.5,
                "geomagnetic_index": 2.0,
            },
            {
                "timestamp": "2025-07-20T12:00:00Z",  # ← latest
                "solar_wind_speed": 500.0,
                "solar_wind_density": 8.0,
                "magnetic_field": 10.0,
                "xray_flux": "2.0e-7",
                "proton_flux": 3.0,
                "geomagnetic_index": 4.0,
            },
            {
                "timestamp": "2025-07-20T11:00:00Z",
                "solar_wind_speed": 420.0,
                "solar_wind_density": 6.0,
                "magnetic_field": 7.5,
                "xray_flux": "1.5e-7",
                "proton_flux": 2.0,
                "geomagnetic_index": 3.0,
            },
        ])

        result = get_current_conditions(dataset_path=csv_file)
        assert result["status"] == "ok"
        assert float(result["observation"]["solar_wind_speed"]) == 500.0


class TestGetCurrentConditionsEdgeCases:
    """Error and edge-case handling."""

    def test_missing_file_returns_error(self, tmp_path):
        result = get_current_conditions(dataset_path=tmp_path / "nonexistent.csv")
        assert result["status"] == "error"
        assert result["observation"] is None
        assert "not found" in result["message"].lower()

    def test_empty_file_returns_error(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        result = get_current_conditions(dataset_path=csv_file)
        assert result["status"] == "error"
        assert result["observation"] is None

    def test_missing_optional_columns_reported(self, tmp_path):
        """A CSV with only timestamp + solar_wind_speed should still succeed
        but report the other expected fields as missing."""
        csv_file = tmp_path / "partial.csv"
        _write_csv(csv_file, [
            {"timestamp": "2025-07-20T10:00:00Z", "solar_wind_speed": 400.0},
        ])
        result = get_current_conditions(dataset_path=csv_file)
        assert result["status"] == "ok"
        assert len(result["missing_fields"]) > 0
        # Fields absent from CSV must be None in the observation
        for field in result["missing_fields"]:
            assert result["observation"][field] is None, (
                f"Expected None for missing field '{field}'"
            )

    def test_observation_fields_are_json_serialisable(self):
        """All values in the observation must survive json.dumps."""
        import json
        result = get_current_conditions()
        assert result["status"] == "ok"
        # Should not raise
        json.dumps(result["observation"])
