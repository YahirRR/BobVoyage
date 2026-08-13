"""
unified_events_loader — converts the official challenge dataset
(space_weather_unified.csv) into the event dict shape expected by
correlate_space_events.py (and by recurrence_forecast.py later).

Drop this into: mcp-server/src/bobvoyage/data/providers/
(alongside local.py, nasa_donki.py, noaa.py — same pattern as the
existing providers)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# event_type mapping: unified CSV labels -> correlate_space_events.py codes
# ---------------------------------------------------------------------------
# correlate_space_events.py only assigns explicit weights to:
#   CME, GST, SEP, FLR, ALERT  (everything else falls back to "OTHER")
# High Speed Stream has no dedicated weight yet -> maps to OTHER for now.
# See note at the bottom of this file if you want to give HSS its own weight.
_EVENT_TYPE_MAP: dict[str, str] = {
    "Solar Flare": "FLR",
    "Geomagnetic Storm": "GST",
    "High Speed Stream": "OTHER",
}


def _safe_str(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return str(val)


def _build_description(row: pd.Series) -> str:
    """Compose a short human-readable description from whatever fields exist."""
    parts: list[str] = []
    if _safe_str(row.get("source_location")):
        parts.append(f"location {row['source_location']}")
    if _safe_str(row.get("active_region")):
        # active_region often comes through as float (e.g. 13664.0)
        ar = row["active_region"]
        ar_str = str(int(ar)) if isinstance(ar, float) and ar.is_integer() else str(ar)
        parts.append(f"active region {ar_str}")
    if _safe_str(row.get("instruments")):
        parts.append(f"observed by {row['instruments']}")
    if _safe_str(row.get("note")):
        parts.append(str(row["note"]))
    return "; ".join(parts) if parts else ""


def load_unified_events(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Load space_weather_unified.csv and return a list of event dicts ready
    to pass as the `events` argument to correlate_space_events().

    Each dict has the shape:
        {
            "event_type":   str,   # FLR | GST | OTHER
            "event_time":   str,   # ISO-8601, UTC
            "external_id":  str,   # original event_id from the CSV
            "source":       str,   # e.g. "NOAA", "unified_dataset"
            "severity":     str | None,  # class_type, e.g. "M2.0", "X1.0", "G0"
            "description":  str,
            # extra fields kept for recurrence_forecast.py / debugging —
            # correlate_space_events.py ignores unknown keys, so these ride along safely
            "active_region": int | None,
            "raw_event_type": str,  # original label before mapping
        }

    Rows with an unparseable begin_time are skipped (mirrors the
    "cannot correlate without a timestamp" behaviour in
    correlate_space_events.py).
    """
    df = pd.read_csv(csv_path)

    # Normalise begin_time to UTC-aware ISO strings
    df["begin_time"] = pd.to_datetime(df["begin_time"], utc=True, errors="coerce")

    events: list[dict[str, Any]] = []
    skipped = 0

    for _, row in df.iterrows():
        if pd.isna(row["begin_time"]):
            skipped += 1
            continue

        raw_type = row["event_type"]
        mapped_type = _EVENT_TYPE_MAP.get(raw_type, "OTHER")

        active_region_val = row.get("active_region")
        active_region: int | None = None
        if active_region_val is not None and not (
            isinstance(active_region_val, float) and pd.isna(active_region_val)
        ):
            try:
                active_region = int(active_region_val)
            except (TypeError, ValueError):
                active_region = None

        source = _safe_str(row.get("source")) or "unified_dataset"

        events.append(
            {
                "event_type": mapped_type,
                "event_time": row["begin_time"].isoformat(),
                "external_id": row["event_id"],
                "source": source,
                "severity": _safe_str(row.get("class_type")),
                "description": _build_description(row),
                "active_region": active_region,
                "raw_event_type": raw_type,
            }
        )

    print(
        f"[unified_events_loader] loaded {len(events)} events, skipped {skipped} (no valid begin_time)"
    )
    return events


if __name__ == "__main__":
    # Quick smoke test
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "space_weather_unified.csv"
    evs = load_unified_events(path)
    print("Sample event:", evs[0])
    print("Total:", len(evs))
    from collections import Counter

    print("By mapped type:", Counter(e["event_type"] for e in evs))
