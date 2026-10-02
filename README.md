# DAFC — Typed Selective Routing

DAFC reuses a GUI action when its successful source outcome and coordinate
transport contract are supported, and otherwise invokes the same target actor.
The maintained interface is the independently installable `dafc` package.

## Install, check and try offline

Use Python 3.12 and uv from the repository root:

```sh
uv sync --locked
uv run --no-sync dafc doctor
uv run --no-sync dafc demo
uv run --no-sync dafc reproduce
uv run --no-sync pytest
```

The demo exercises geometric action transfer and failed-source fallback with a
fixed seed. It needs no model, browser, network or emulator. `reproduce` recomputes
the public historical estimates and their 20,000-sample bootstrap intervals.

## Explicit runs

```sh
uv run --no-sync python examples/make_routing_inputs.py --directory "$HOME/research/data/DAFC/example"
uv run --no-sync dafc doctor --config configs/routing.yaml
uv run --no-sync dafc run --config configs/routing.yaml
```

Choose a fresh `run_id` for another experiment. The sample configuration places
outputs under `~/research/runs/DAFC/`. It can be copied anywhere: its relative paths
resolve from the configuration file, never from the invoking shell's directory.
See [resources](docs/RESOURCES.md) before configuring a real browser or Android
episode, atomic batch, model service or learned matcher. Live runs are explicit;
installation and the four basic checks do not download models or datasets.

## Research scope and evidence

The sealed C587-v3 Android diagnostic contains 1120 jobs and one additional
retained infrastructure attempt. Both actors used 25% fewer target model calls
under DAFC: Qwen had 120/140 successes (always: 119/140), and UI-TARS 100/140
(always: 99/140). This fixed seven-task action-level diagnostic does not establish
AndroidWorld-wide performance. Browser studies and full raw evidence are external.

Current runs have a new source identity and `paper_evidence: false`. Original
evidence, checksums and source are retained byte-for-byte; see
[migration](docs/MIGRATION.md) and [validation](docs/VALIDATION.md).

The [architecture](docs/ARCHITECTURE.md) separates routing, transition state,
platform interaction, model adapters, run ownership and evidence. Predecessor
registration layers remain only where current algorithms depend on them.
Owned code is Apache-2.0; external resources retain their own terms.
See [contributing](CONTRIBUTING.md) and [release notes](docs/RELEASE_NOTES.md).
