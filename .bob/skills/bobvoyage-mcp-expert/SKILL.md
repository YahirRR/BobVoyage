---
name: bobvoyage-mcp-expert
description: Expert guidance for working on the BobVoyage space weather intelligence MCP server. Use this skill whenever adding, modifying, or debugging tools in src/bobvoyage/tools/, changing server.py, writing tests, or extending the dashboard. Covers project conventions, tool architecture, and rules that prevent common bugs seen in this codebase (wrong dataset paths, causal language leaking into risk assessments, keyword substring collisions, incorrect field reads from nested output shapes).
---

# BobVoyage Space Weather Intelligence Copilot

BobVoyage is an MCP (Model Context Protocol) server that turns NASA/NOAA space weather
data into mission-relevant intelligence: anomaly detection, forecasting, risk scoring,
event correlation, active-region recurrence projection, and audience-specific briefings.
Built for the IBM AI Builders Challenge — Space Exploration theme.

## When to use this skill

Use this skill for any task that touches:
- `src/bobvoyage/tools/*.py` — the 8 MCP tools
- `src/bobvoyage/server.py` — MCP tool registration
- `src/bobvoyage/dashboard/*` — the web dashboard and its conversational chat
- `tests/*` — anything under the test suite
- `data/space_weather_unified.csv` or `data/space_weather.csv` — the two datasets

## The two datasets — do not confuse them

| File | Shape | Used by |
|---|---|---|
| `data/space_weather_unified.csv` | 1,743 discrete events (Solar Flare, High Speed Stream, Geomagnetic Storm), 2-year span, columns include `active_region`, `source_location`, `class_type` | `recurrence_forecast.py`, event correlation input (via `unified_events_loader.py`) |
| `data/space_weather.csv` | 200 rows of continuous telemetry, 5-min interval, columns: `solar_wind_speed`, `solar_wind_density`, `magnetic_field`, `xray_flux`, `proton_flux`, `geomagnetic_index` | `predict_conditions.py`, `detect_anomalies.py`, `analyze_trends.py`, `get_current_conditions.py` |

Before writing code that reads either file, confirm which one the task actually needs —
mixing them up is a recurring failure mode in this codebase.

## Tool inventory and typical call order

1. `get_current_conditions` — latest telemetry snapshot
2. `analyze_trends` — direction/severity of recent parameter changes
3. `detect_anomalies` — z-score based deviation from baseline
4. `predict_conditions` — Holt Double Exponential Smoothing forecast
5. `correlate_space_events` — temporal association between DONKI/NOAA events and telemetry (never causal)
6. `recurrence_forecast` — active-region re-entry risk via heliographic position + solar rotation rate (NOT via active_region ID persistence — verified empirically not to hold in this dataset)
7. `assess_mission_risk` — combines 1–6 into domain-level risk scores (radiation, communications, navigation, power, attitude_control)
8. `stakeholder_briefing` — translates the output of #7 into audience-specific guidance (satellite_operator, astronaut, aviation, power_grid)

A full pipeline typically calls tools 1–6 to gather evidence, then 7, then optionally 8.
No tool duplicates another's responsibility — each file's docstring has an explicit
"Responsibility: X only" line. Preserve this separation when editing.

## Non-negotiable conventions

**Error shape.** Every tool returns `{"status": "ok" | "error", ..., "message": str}`
on both success and failure. Never raise an unhandled exception from a public function;
wrap and return the standard error shape via a local `_error()` helper.

**Epistemic labels.** Evidence strings must be prefixed with their source category:
`OBSERVED:` (raw data), `ANALYZED:` (derived trend/anomaly), `PREDICTED:` (forecast),
`CORRELATED:` (event association), `PROJECTED:` (recurrence forecast). Never present
a prediction or correlation as an observed fact.

**No causal language, ever.** Every module that touches correlation or risk uses a
`causal_guard()` / `_causal_guard_risk()` function that strips forbidden phrases
("caused", "due to", "resulted in", "led to", "responsible for", etc.) and replaces
them with neutral temporal-association language. Any new evidence-generating code
must run through the same guard. `correlate_space_events` only ever reports
`no_significant_correlation` / `weak_temporal_association` / `moderate_temporal_association`
/ `strong_temporal_association` — never causation.

**Docstring structure.** New tools follow the existing two-section pattern:
`SCIENTIFIC CONSTRAINT` (what NOT to claim, and why, with empirical justification
if a naive approach was rejected) followed by `METHODOLOGY` (the actual math/logic,
numbered). Look at `recurrence_forecast.py` or `assess_mission_risk.py` for the
reference format before writing a new tool.

**Risk/interpretation labels are fixed vocabularies.** Don't invent new label sets.
Reuse: `LOW/MODERATE/HIGH/CRITICAL` (mission risk), `low_recurrence_risk/
moderate_recurrence_risk/elevated_recurrence_risk` (recurrence), the four
`*_temporal_association` labels (correlation).

## Known failure modes — check for these before considering a task done

1. **Nested vs. flat field reads.** `correlate_space_events()` returns event type at
   `corr["event"]["type"]`, not `corr["event_type"]`. A past bug read the wrong flat
   key and silently fell back to `"OTHER"` while evidence text (built correctly
   elsewhere) said the real type. When reading another tool's output, verify the
   actual nested shape by reading that tool's source — don't assume a flat dict.

2. **Keyword substring collisions.** Naive `if keyword in text` matching can false-positive
   on substrings (e.g. `"eva"` matching inside `"Elevated"`). When building keyword-based
   routing or filtering (chat intent detection, audience keyword matching), test against
   real sentence fragments from actual tool output, not just the keyword in isolation.

3. **General/fallback content overshadowing specific content.** In
   `stakeholder_briefing.py`, "general" recommendations were once prepended
   unconditionally, making all 4 audiences look identical. Fallback/general content
   must only appear when there is truly zero specific match — never prepended
   alongside specific matches.

4. **Dataset path resolution.** `_DEFAULT_DATASET` / `_PROJECT_ROOT` constants use
   `Path(__file__).resolve().parents[N]`. Before trusting a default path, check it
   against the actual repo structure — the correct `N` depends on how many directories
   sit between the tool file and the project root, which has been a source of bugs
   when files move.

5. **Dependency version drift.** `mcp` package version must stay pinned
   `>=1.0.0,<2.0.0` in `requirements.txt`. Version 2.0.0 replaced the
   `@app.list_tools()` / `@app.call_tool()` decorator API used throughout
   `server.py` with a constructor-based `on_list_tools=` pattern, breaking the
   server on import. If you see `AttributeError: 'Server' object has no attribute
   'list_tools'`, this is the cause — check the installed version before debugging
   further.

## Testing conventions

- Test files mirror tool files: `tests/test_<tool_name>.py`.
- Group tests into classes by behavior category (`TestOutputSchema`,
  `TestErrorHandling`, `TestDeterminism`, `TestInternalHelpers`, `TestEndToEnd`, etc.),
  matching the structure already used across the suite.
- Every new tool needs, at minimum: schema validation, nominal-case behavior,
  error/invalid-input handling, determinism (same input → same output), and
  JSON-serializability of the full output.
- When a bug is found and fixed, add a test that pins the specific bug (not just a
  broad "it works" test) so it cannot silently regress.
- Run the full suite after any change, not just the new test file, to catch
  cross-tool regressions: `pytest tests/ -v`.

## Before starting any task

1. Confirm which dataset the task needs (see table above).
2. Read the target tool's existing docstring and its test file before editing.
3. If adding a new tool, read at least one existing tool of similar complexity as
   a style reference — do not invent a new structure.
4. If the environment fails to run pytest with `ModuleNotFoundError: No module
   named 'bobvoyage'`, the project needs `pip install -e .` from the directory
   containing `pyproject.toml` — this is an environment issue, not a code issue.
