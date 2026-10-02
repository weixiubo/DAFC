# Validation — 2026-10-02

Baseline public revision: `8b0eae11468195ce86ccb33fea980334c72ac2d6`.

## Executed successfully

- 108 tests and five subtests, including the original 22 scenarios. The test
  mapping keeps all original assert expressions and unittest assertions; six
  call expressions change only the module owning the same tested function.
- Exact seeded routing receipts for identity, translation and blank targets.
- Twenty original/current browser and Android controller comparisons, sixteen
  transition-selector comparisons and ten atomic-trial comparisons. These
  compare event order, state/progress, actions, model-call budgets, retry/failure
  classification and cleanup with deterministic platform stand-ins. Original
  function bodies come from the hash-verified reference fixtures.
- Model-service backend isolation, local-only model loading, checkpoint content
  hashes, matcher source/weight identity, explicit configuration precedence,
  missing resources, malformed inputs, result tampering and ambiguous recovery.
- Ruff formatting/lint, strict mypy on five configuration/resource/run-ownership
  boundaries, and 600-line / 120-line / complexity-15 structure checks.
- A wheel and sdist include the Apache-2.0 license and required public evidence.
  An independent wheel environment outside the source tree imports every module,
  runs doctor/demo, executes real small-image routing and recomputes historical
  estimates. Imports and diagnostics create no caller or user resource folders.
- A checksum-controlled source export builds a new `.venv` from uv.lock and runs
  the complete suite outside the original checkout.
- Public-file and proposed-history credential-pattern scan, manifest export and
  byte comparison of all original reproduce/ files. Pattern scans are a bounded
  check, not proof that every possible private datum can be detected.

## Historical frozen results

The 1120 published outcomes, eight policy summaries and six paired comparisons
recompute with 20,000 bootstrap samples. Original evidence and checksum bytes
are unchanged. The original complete source is separately hash-bound. New
experiments explicitly set `paper_evidence: false` and have new run identities.

## Resource-dependent checks outstanding

Full raw-image/attempt re-verification, GPU numerical inference, actual browser
and Android execution remain pending restoration. Controller stand-ins do not
establish real-device equivalence. The Python 3.12 Android dependency group differs
from the old Python 3.11 runtime; that integration is declared and locked but
unverified. Model checkpoints, large task data and emulator assets were not
downloaded automatically. No paid model or remote experiment service was called.

The material-cleanup checks and current retained source list are documented in
MATERIALS.md and tests/reference/MANIFEST.json. Whole-source bundles are no longer
part of the current distribution. Scientific evidence identities are unchanged.

## Final distribution audit — 2026-10-03

The current suite passed 108 collected cases. The five DAFC subtests also passed. The source distribution was unpacked into a new directory and installed in its own new locked environment.
All required test fixtures and helper files were present. The rebuilt wheel
payload matched the wheel built from the maintained checkout. Installed commands
ran outside the source directory with their declared resource and optional
dependency checks. Lint, formatting, types, structural limits and lock consistency
passed. CI now includes `check_sdist.py` after the ordinary build.

This verifies the maintained offline software and distribution boundaries.
The resource-dependent experiments listed above remain unverified; the test
result does not imply a new GPU experiment or a changed scientific conclusion.
