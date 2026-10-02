# Contributing

Use this repository's `.venv`: `uv sync --locked`. Do not use a sibling project's
environment, a restored `.venv`, global Python packages or hard-coded home paths.

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync python scripts/check_structure.py
uv run --no-sync pytest
uv run --no-sync dafc reproduce
uv build --out-dir dist
uv run --no-sync python scripts/check_release.py
```

Use explicit types at configuration and run-state boundaries. Keep maintained
modules at most 600 lines, functions at most 120 lines and complexity at most 15.
Frozen statistical code is byte-checked, not formatted or rewritten. There are
no ordinary-code structure exceptions. Preserve original test scenarios and add
behavioral checks when changing state transitions or scientific decisions.

For a release, review and regenerate `PUBLIC_MANIFEST.json`, then run
`python scripts/export_public.py --target /path/to/new/export`. The exporter
rejects changed bytes and unsafe paths. Run the documented checks in that export
with a new environment before publishing. Do not export private evidence, model
weights, service audit files, credentials or virtual environments.
