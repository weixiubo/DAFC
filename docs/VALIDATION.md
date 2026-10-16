# Validation

Run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md) using the locked
Python 3.12 environment.

| Check | Coverage |
| --- | --- |
| Main suite | 119 tests and five subtests, including the original 22 scenarios |
| Controller comparisons | 20 deterministic browser/Android cases |
| Selector comparisons | 16 transition-selection cases |
| Atomic trials | 10 cases covering decisions, retries, outcomes and cleanup |
| Public boundaries | Configuration precedence, resource hashes, service isolation, checkpoint identities and recovery |
| Engineering checks | Ruff, five strict type boundaries, 600-line modules, 120-line functions and McCabe complexity 15 |

Controller comparisons use deterministic platform stand-ins and compare
event order, progress, actions, call budgets and failure classification.
Reference function bodies are verified against their producer hashes.
`TEST_MIGRATION.json` records assertion and ownership mappings.

## Evidence verification

`dafc reproduce` processes 1,120 published outcomes, eight policy summaries
and six paired comparisons using 20,000 bootstrap samples and the recorded
seeds. `--raw-root PATH` additionally validates original attempts and image
receipts. See the [reproduction guide](../reproduce/README.md).

## Distribution verification

Wheel checks import installed modules and execute diagnostics, image routing
and statistical reproduction outside the source tree. Source-distribution
checks create a fresh locked environment, run the full suite and compare
rebuilt wheel contents. `PUBLIC_MANIFEST.json` authenticates source-export
files. Platform execution uses the separate setups in [RESOURCES.md](RESOURCES.md).

## Artifact restoration checks

Six artifact restoration tests cover Unicode paths, missing and corrupt parts,
repeat execution, modified-file protection, upstream assets and optional
models. The suite counts above include these cases. The restoration and
research-material checks performed on 2026-10-15 are recorded in the
[research archive](https://github.com/weixiubo/DAFC/releases/tag/handoff-2026-10-15).
