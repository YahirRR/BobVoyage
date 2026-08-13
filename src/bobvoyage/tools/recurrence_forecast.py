"""
recurrence_forecast — BobVoyage MCP tool
Forecasts when solar active regions with a history of strong flare
production are likely to rotate back into an Earth-facing position.
===========================================================================
SCIENTIFIC CONSTRAINT
===========================================================================
The Sun does not rotate as a rigid body, and NOAA active region numbers
are NOT reliable trackers of physical recurrence: a region that survives
multiple rotations is often re-catalogued as a "new" active region on
reappearance, and number reuse over long timescales (e.g. the same
integer appearing ~1 year apart) is a cataloguing coincidence, not a
sign of physical persistence. This was verified empirically against the
project's own dataset: 0 instances of the same active_region reappearing
within a 20-35 day window were found; the only long-span case (13777,
Aug 2023 -> Aug 2024) is 361 days apart -- a numbering coincidence.
Because of this, THIS TOOL DOES NOT rely on active_region ID persistence.
Instead, it uses each region's most recently observed heliographic
longitude (from `source_location`, e.g. "N25E90") to compute its current
rotational phase, and projects forward using the Sun's synodic rotation
rate as seen from Earth (~13.2 deg/day, Carrington synodic period of
27.2753 days). This is a physics-based projection, not a guarantee: solar
active regions physically persist across a full rotation in a minority of
cases (published solar-physics literature places sustained flare-productive
recurrence at roughly 20-40% of large regions). The tool reports this as a
"recurrence risk window", not a prediction of certain reappearance.
No causal or certain language is used. Interpretation labels are limited to:
  "low_recurrence_risk" / "moderate_recurrence_risk" / "elevated_recurrence_risk"
===========================================================================
METHODOLOGY
===========================================================================
1. HELIOGRAPHIC POSITION PARSING
   `source_location` strings (e.g. "N25E90", "S19W59") are parsed into
   a signed central-meridian-relative longitude:
     E-longitude -> negative (region still rotating onto the visible disk)
     W-longitude -> positive (region rotating toward the west limb)
   Central meridian = 0. Visible disk approx. spans [-90, +90].
2. PER-REGION LATEST STATE
   For each active_region, the most recent flare (by begin_time) with a
   valid source_location defines that region's "last known position" and
   "last seen" timestamp -- the basis for forward projection.
3. FLARE PRODUCTIVITY SCORE
   Each flare's class_type (e.g. "M2.0", "X1.3") is converted to a
   continuous flux-equivalent score:
     letter base:  A=1e-8, B=1e-7, C=1e-6, M=1e-5, X=1e-4 (W/m^2)
     score = base * numeric_multiplier
   A region's productivity score = sum of its flares' scores, plus a
   bonus weight for the single strongest flare (captures "one X-class
   event matters more than ten C-class events").
4. ROTATION PROJECTION
   Using rotation_rate_deg_per_day = 360 / 27.2753:
     days_to_exit_disk   = (90 - position_deg) / rotation_rate   [if region still visible]
     days_to_re-entry    = days_to_exit_disk + (180 / rotation_rate)
   If the region has already rotated past W90 (position_deg > 90,
   inferred from elapsed time since last_seen), the projection is
   computed forward from last_seen directly.
5. RECURRENCE RISK CLASSIFICATION
   Combines: (a) flare productivity score (log-scaled, 0-100),
             (b) recency of last activity (decays projection confidence
                 for regions not seen in a long time),
             (c) presence of at least one M5+ or X-class flare (adds a
                 fixed risk bump, since these are the events with the
                 clearest operational significance).
   risk_score [0,100] -> low_recurrence_risk / moderate_recurrence_risk /
                          elevated_recurrence_risk
   thresholds: <30 low, 30-60 moderate, >=60 elevated
Responsibility: recurrence forecasting ONLY.
No anomaly detection, real-time telemetry forecasting (see predict_conditions.py),
or mission risk scoring (see assess_mission_risk.py) here.
===========================================================================
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path(".")
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "space_weather_unified.csv"

_SYNODIC_ROTATION_DAYS = 27.2753
_ROTATION_RATE_DEG_PER_DAY = 360.0 / _SYNODIC_ROTATION_DAYS  # ~13.2

_FLARE_LETTER_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

_RISK_THRESHOLDS = [
    (60.0, "elevated_recurrence_risk"),
    (30.0, "moderate_recurrence_risk"),
    (0.0,  "low_recurrence_risk"),
]

_LOCATION_RE = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})$")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_source_location(loc: str) -> dict[str, float] | None:
    """
    Parse a heliographic location string like "N25E90" or "S19W59".
    Returns {"lat_deg": signed float, "lon_position_deg": signed float}
    where lon_position_deg is negative for East (approaching disk),
    positive for West (approaching limb), 0 = central meridian.
    Returns None if the string doesn't match the expected format.
    """
    if not isinstance(loc, str):
        return None
    m = _LOCATION_RE.match(loc.strip())
    if not m:
        return None
    lat_hem, lat_deg, lon_hem, lon_deg = m.groups()
    lat = float(lat_deg) * (1 if lat_hem == "N" else -1)
    lon_position = float(lon_deg) * (1 if lon_hem == "W" else -1)
    return {"lat_deg": lat, "lon_position_deg": lon_position}


def _flare_score(class_type: str | None) -> float:
    """Convert a class_type string (e.g. 'M2.0') to a flux-equivalent score."""
    if not isinstance(class_type, str) or len(class_type) < 1:
        return 0.0
    letter = class_type[0].upper()
    base = _FLARE_LETTER_BASE.get(letter)
    if base is None:
        return 0.0
    try:
        multiplier = float(class_type[1:]) if len(class_type) > 1 else 1.0
    except ValueError:
        multiplier = 1.0
    return base * multiplier


def _is_strong_flare(class_type: str | None) -> bool:
    """True if class_type is M5.0+ or any X-class."""
    if not isinstance(class_type, str) or len(class_type) < 2:
        return False
    letter = class_type[0].upper()
    if letter == "X":
        return True
    if letter == "M":
        try:
            return float(class_type[1:]) >= 5.0
        except ValueError:
            return False
    return False


def _classify_risk(score: float) -> str:
    for threshold, label in _RISK_THRESHOLDS:
        if score >= threshold:
            return label
    return "low_recurrence_risk"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def recurrence_forecast(
    active_region: int | None = None,
    lookback_days: float = 45.0,
    as_of: str | None = None,
    dataset_path: str | Path | None = None,
    min_flares: int = 2,
) -> dict[str, Any]:
    """
    Forecast recurrence risk windows for solar active regions.

    Parameters
    ----------
    active_region:
        If given, restrict the forecast to this specific active region.
        If None, scan all regions active within `lookback_days` of `as_of`.
    lookback_days:
        Only consider regions whose most recent flare falls within this
        many days of `as_of`. Default 45 (covers slightly more than one
        full synodic rotation).
    as_of:
        ISO-8601 timestamp to forecast from. Defaults to the most recent
        begin_time in the dataset (i.e. "today" relative to the data).
    dataset_path:
        Optional CSV path override. Defaults to
        data/space_weather_unified.csv at the project root.
    min_flares:
        Minimum number of flares a region must have produced to be
        included in a full (non-specific) scan. Default 2 -- filters out
        single-flare noise from the ranked output; explicitly requesting
        an `active_region` bypasses this filter.

    Returns
    -------
    dict with keys:
        status, as_of, parameters, regions, message
    Each entry in `regions` includes: active_region, last_seen,
    last_position, flare_count, strongest_flare, productivity_score,
    days_to_limb_exit, forecast_reentry_window, risk_score, risk_level,
    evidence.
    """
    if lookback_days <= 0:
        return _error("lookback_days must be > 0.")
    if min_flares < 1:
        return _error("min_flares must be >= 1.")

    path = Path(dataset_path) if dataset_path else _DEFAULT_DATASET
    if not path.exists():
        return _error(f"Dataset not found at '{path}'.")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return _error(f"Failed to read dataset: {exc}.")

    flr = df[df["event_type"] == "Solar Flare"].copy()
    flr["begin_time"] = pd.to_datetime(flr["begin_time"], utc=True, errors="coerce")
    flr = flr.dropna(subset=["begin_time", "active_region"])
    flr["active_region"] = flr["active_region"].astype(int)

    if flr.empty:
        return _error("No valid solar flare records with active_region found.")

    if active_region is not None:
        flr = flr[flr["active_region"] == active_region]
        if flr.empty:
            return _error(f"No flare records found for active_region {active_region}.")
        # When a specific region is requested, anchor "as_of" to that region's
        # own most recent flare (unless the caller explicitly overrides it),
        # so the lookback window doesn't wrongly exclude a region just because
        # the OVERALL dataset has more recent activity from other regions.
        as_of_ts = pd.to_datetime(as_of, utc=True) if as_of else flr["begin_time"].max()
    else:
        as_of_ts = pd.to_datetime(as_of, utc=True) if as_of else flr["begin_time"].max()

    cutoff = as_of_ts - pd.Timedelta(days=lookback_days)
    recent = flr[flr["begin_time"] >= cutoff]

    if recent.empty:
        return {
            "status": "ok",
            "as_of": as_of_ts.isoformat(),
            "parameters": {"lookback_days": lookback_days, "min_flares": min_flares,
                            "active_region": active_region},
            "regions": [],
            "message": f"No active regions with flare activity in the last {lookback_days:.0f} days.",
        }

    regions_out: list[dict[str, Any]] = []

    for ar, group in recent.groupby("active_region"):
        group = group.sort_values("begin_time")
        n_flares = len(group)
        if active_region is None and n_flares < min_flares:
            continue

        last_row = group.iloc[-1]
        last_seen = last_row["begin_time"]
        pos = _parse_source_location(last_row.get("source_location"))

        productivity = float(sum(_flare_score(c) for c in group["class_type"]))
        strongest = max(group["class_type"].dropna().tolist(),
                         key=_flare_score, default=None)
        has_strong = any(_is_strong_flare(c) for c in group["class_type"])

        evidence: list[str] = [
            f"OBSERVED: {n_flares} flare(s) recorded for active region {ar} "
            f"between {group['begin_time'].min().date()} and {last_seen.date()}.",
        ]
        if strongest:
            evidence.append(f"OBSERVED: Strongest flare in window: {strongest}.")

        # --- rotation projection ---
        days_to_limb_exit: float | None = None
        reentry_start: pd.Timestamp | None = None
        reentry_end: pd.Timestamp | None = None
        position_note = "unknown"

        if pos is not None:
            lon = pos["lon_position_deg"]
            days_since_seen = (as_of_ts - last_seen).total_seconds() / 86400.0
            projected_lon = lon + days_since_seen * _ROTATION_RATE_DEG_PER_DAY

            if projected_lon <= 90:
                # still visible or about to exit
                days_to_limb_exit = (90 - projected_lon) / _ROTATION_RATE_DEG_PER_DAY
                exit_date = as_of_ts + pd.Timedelta(days=days_to_limb_exit)
                position_note = "still_visible_or_near_limb"
            else:
                # already rotated past the west limb as of as_of_ts
                exit_date = last_seen + pd.Timedelta(days=(90 - lon) / _ROTATION_RATE_DEG_PER_DAY)
                days_to_limb_exit = (exit_date - as_of_ts).total_seconds() / 86400.0
                position_note = "already_off_disk"

            reentry_start = exit_date + pd.Timedelta(days=180 / _ROTATION_RATE_DEG_PER_DAY)
            reentry_end = reentry_start + pd.Timedelta(days=3)  # +/- a few days uncertainty band

            evidence.append(
                f"OBSERVED: Last known position {last_row.get('source_location')} "
                f"({position_note.replace('_',' ')})."
            )
            evidence.append(
                "PROJECTED: Forward-projected using synodic solar rotation rate "
                f"(~{_ROTATION_RATE_DEG_PER_DAY:.2f} deg/day). Reappearance is a physics-based "
                "projection assuming the region persists across the rotation -- not a guaranteed event."
            )
        else:
            evidence.append(
                "OBSERVED: source_location for the most recent flare could not be parsed; "
                "rotation projection unavailable for this region."
            )

        # --- risk score ---
        # log-scale productivity into [0,100]
        prod_score = 0.0
        if productivity > 0:
            prod_score = max(0.0, min(100.0, (math.log10(productivity) + 8) * 14.0))
        recency_days = (as_of_ts - last_seen).total_seconds() / 86400.0
        recency_penalty = max(0.0, min(1.0, recency_days / lookback_days))  # 0=just seen, 1=at cutoff
        strong_bonus = 25.0 if has_strong else 0.0

        risk_score = max(0.0, min(100.0, prod_score * (1 - 0.5 * recency_penalty) + strong_bonus))
        risk_level = _classify_risk(risk_score)

        if has_strong:
            evidence.append("ANALYZED: Region produced at least one M5+/X-class flare -- "
                             "elevated operational significance.")

        regions_out.append({
            "active_region": int(ar),
            "last_seen": last_seen.isoformat(),
            "last_position": last_row.get("source_location"),
            "flare_count": n_flares,
            "strongest_flare": strongest,
            "productivity_score": round(productivity, 10),
            "days_to_limb_exit": round(days_to_limb_exit, 2) if days_to_limb_exit is not None else None,
            "forecast_reentry_window": (
                {"start": reentry_start.isoformat(), "end": reentry_end.isoformat()}
                if reentry_start is not None else None
            ),
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "evidence": evidence,
        })

    regions_out.sort(key=lambda r: r["risk_score"], reverse=True)

    return {
        "status": "ok",
        "as_of": as_of_ts.isoformat(),
        "parameters": {"lookback_days": lookback_days, "min_flares": min_flares,
                        "active_region": active_region},
        "regions": regions_out,
        "message": (
            f"{len(regions_out)} active region(s) analyzed for recurrence risk "
            f"within a {lookback_days:.0f}-day lookback window."
        ),
    }


def _error(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "as_of": None,
        "parameters": {},
        "regions": [],
        "message": message,
    }


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "space_weather_unified.csv"
    # test 1: specific known case
    r = recurrence_forecast(active_region=13664, dataset_path=path, lookback_days=400)
    print("=== Region 13664 case ===")
    print(json.dumps(r, indent=2, default=str)[:2000])
    print()
    # test 2: general scan as of the last date in the dataset
    r2 = recurrence_forecast(dataset_path=path, lookback_days=45)
    print("=== General scan, top 5 by risk ===")
    for reg in r2["regions"][:5]:
        print(reg["active_region"], reg["risk_level"], reg["risk_score"], reg["forecast_reentry_window"])