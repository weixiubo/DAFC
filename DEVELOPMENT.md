# Development

Use Python 3.12 with the repository's `uv.lock` and project-local `.venv`.
`uv sync --locked` installs the routing implementation and development tools.
Optional groups provide browser, Android, inference and learned-matcher
dependencies.

Source resides in `~/project/DAFC`; data, models and outputs use configurable
research roots. See [resources](docs/RESOURCES.md) for path resolution and the
Python 3.11 environment recorded for the original Android experiment.
[Contributing](CONTRIBUTING.md) lists the complete validation commands.
