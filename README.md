# DAFC — Typed Selective Routing

DAFC transfers a GUI action when its successful source outcome and coordinate
transport contract support reuse. Otherwise, it invokes the target actor.
The `dafc` package provides routing, browser and Android execution interfaces,
model adapters and evidence verification.

## Research artifacts

The [artifact guide](docs/ARTIFACTS.md) describes the recorded experiments,
manuscript sources, datasets and historical source versions in the
[research archive](https://github.com/weixiubo/DAFC/releases/tag/research-artifacts-2026-10-17).
Downloads are public and include file-level integrity verification.

## Installation and example

Use Python 3.12 and uv from the repository root:

```sh
uv sync --locked
uv run --no-sync dafc doctor
uv run --no-sync dafc demo
uv run --no-sync dafc reproduce
```

The deterministic demo exercises geometric action transfer and fallback after
a failed source action. It uses small offline inputs. `reproduce` rebuilds
the published estimates and their 20,000-sample bootstrap intervals.

## Configured execution

```sh
uv run --no-sync python examples/make_routing_inputs.py --directory "$HOME/research/data/DAFC/example"
uv run --no-sync dafc doctor --config configs/routing.yaml
uv run --no-sync dafc run --config configs/routing.yaml
```

Assign a distinct `run_id` to each experiment. The example writes to
`~/research/runs/DAFC/`. Relative input paths resolve from the configuration
file, so the configuration can be invoked from another working directory.

Browser episodes, Android episodes and atomic batches use explicit platform,
model-service and matcher resources. [Resources](docs/RESOURCES.md) describes
the optional dependency groups, configuration templates and preflight checks.

## Research evidence

The C587-v3 Android study is a seven-task, action-level diagnostic with
1,120 jobs and one retained infrastructure retry. DAFC used 25% fewer target
model calls for each actor. Qwen achieved 120/140 successes, compared with
119/140 under the always-call control; UI-TARS achieved 100/140, compared
with 99/140. The [reproduction guide](reproduce/README.md) describes the
published outcome records, statistical analysis and original-attempt checks.

Each configured run records its source, inputs, configuration and service
identities. The original study records retain their producer identities.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [repository layout](docs/MATERIALS.md)
- [Development](DEVELOPMENT.md) and [contributing](CONTRIBUTING.md)
- [Migration](docs/MIGRATION.md), [release notes](docs/RELEASE_NOTES.md) and [validation](docs/VALIDATION.md)
- [Third-party notices](THIRD_PARTY.md)

Project code is licensed under Apache-2.0.
