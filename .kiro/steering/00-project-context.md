# Project Context

Last reviewed: 2026-08-17

- `solar-lens` is an AGPL-3.0 Home Assistant integration/add-on prototype for solar battery prediction; README currently labels it early development and not functional.
- Python package metadata requires Python >=3.14; runtime code is under `custom_components/solar_lens/` and tests under `tests/`.
- Planned architecture combines consumption and solar ML models with a physical battery simulator for a 48-hour SoC forecast, plus sensors for empty time, charge resume, remaining hours, gap, and prediction curve.
- Development dependencies and quality gates are defined in `pyproject.toml`: pytest with 80% coverage, Ruff, strict mypy, and Bandit.
- Do not treat `solar-lens-backup` as the current implementation; it is a separate legacy add-on copy.
