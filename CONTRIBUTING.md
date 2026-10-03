# Contributing

Create the project-local Python 3.12 environment with `uv sync --locked`.

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync python scripts/check_structure.py
uv run --no-sync pytest
uv run --no-sync dafc reproduce
uv build --out-dir dist
uv run --no-sync python scripts/check_release.py
uv run --no-sync python scripts/check_sdist.py
```

Use explicit types at configuration and run-state boundaries. Maintained
modules have a 600-line limit, functions a 120-line limit and McCabe complexity
a limit of 15. Add behavioral tests when changing routing decisions, state
transitions or platform recovery. Original statistical code and regression
references retain their SHA-256 identities.

Keep complete datasets, model weights, credentials and private run records
in configured external locations. Before export, review the file set, update
`PUBLIC_MANIFEST.json` and run `scripts/export_public.py --target PATH` with a
new target directory. Release checks validate installed-wheel behavior; the
sdist check runs the full suite in a fresh locked environment and compares
rebuilt wheel contents. Cache development dependencies before this offline
distribution check.
