# DAFC — Typed Selective Routing

Reuse a GUI action only when its successful source outcome and its coordinate
transport contract are both supported. Otherwise use the same target actor.
The maintained implementation is the `dafc` package under `src/`; historical
script names are thin compatibility entry points.

## Install and verify

Python 3.12 is the development target. From this repository:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python examples/routing_smoke.py
.venv/bin/python -m pytest
.venv/bin/python reproduce/verify.py
```

The smoke example makes no model, browser, network or emulator calls. Real
execution requires the explicitly supplied resources in `docs/RESOURCES.md`.
Run `scripts/android_atomic_closed_loop.py --help` for the retained arguments.

## Evidence status

The original C587-v3 Android diagnostic completed **1120 jobs**, with one
additional retained infrastructure attempt. Both actors used 25% fewer target
model calls under DAFC. Qwen had 120/140 successes (always:119/140), and UI-TARS
100/140 (always:99/140). This is a fixed seven-task, action-level diagnostic;
it does not establish AndroidWorld-wide performance. Older notes reporting
51/1120 are obsolete. Browser V5 evidence and exploratory studies are retained
in the private research archive and listed as external material.

`reproduce/verify.py` rebuilds summaries and paired task-cluster bootstrap
intervals without models. See `reproduce/README.md` for the raw evidence boundary.

## Layout

* `dafc.registration`, `routing`, `coordinates`: image transport and fallback.
* `dafc.evidence`, `storage`: contracts, append-only records and immutable writes.
* `dafc.models`: model/service clients and retained baseline implementations.
* `dafc.adapters`: Android and browser interaction.
* `dafc.experiments`, `evaluation`: orchestration and statistics.

Necessary predecessor registration implementations remain because V6 uses
their fitting and selection primitives. They are not separate maintained methods.
Owned code is Apache-2.0. External runtimes/models keep their own licenses and
are not copied into this repository. Export only the reviewed public manifest:
`python scripts/export_public.py --target /path/to/new/export`.
