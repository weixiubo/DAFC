# Resource and environment contract

Defaults are `~/research/data/DAFC`, `~/research/models`, `~/research/runs/DAFC`
and `~/research/data/<project>/evidence`. Resolution precedence is command flags / explicit YAML
paths, environment variables, user configuration, then defaults. Flags override
the YAML paths. Environment variables are DAFC_DATA_DIR, DAFC_MODELS_DIR,
DAFC_RUNS_DIR and DAFC_EVIDENCE_DIR. User configuration is
`$XDG_CONFIG_HOME/research/DAFC.json` (default `~/.config/research/DAFC.json`):

```json
{"paths": {
  "data": "/path/to/external/data",
  "models": "/path/to/external/models",
  "runs": "/path/to/external/runs",
  "evidence": "/path/to/external/data/DAFC/evidence"
}}
```
Set DAFC_LOCAL_CONFIG to select another local configuration file. Individual
catalog resources can be mapped in a `resources` object in that JSON file or
through DAFC_RESOURCE_<UPPERCASE_ID_WITH_UNDERSCORES>.

YAML input paths accept {data}, {models}, {runs} and {evidence}. Relative values
resolve from their configuration file. `doctor` is read-only; missing optional
resources are reported with retrieval instructions. `doctor --require ID` makes
a particular catalog item mandatory. `doctor --config FILE` checks all declared
input hashes, workflow requirements and optional dependencies before a run.

The packaged resource_catalog.json records each purpose, source, version,
license boundary, location, checksum where known and specific restoration steps.
An unknown hash/revision is explicitly unverified, not silently trusted. Public
statistical inputs ship with the wheel; large resources and original raw images
do not. `dafc reproduce --raw-root PATH` verifies the original C587-v3 manifest
and image receipts after deliberate restoration.

## Optional environments

| Process | Install group and external requirement |
| --- | --- |
| Basic routing, testing, public evidence | uv sync --locked |
| Browser episodes | uv sync --locked --extra browser; matching Chrome, pinned driver and task server |
| Android episodes/batches | uv sync --locked --extra android; pinned AndroidWorld source, emulator, SDK and APK |
| Model process | uv sync --locked --extra inference; explicit local checkpoint |
| Learned matcher process | uv sync --locked --extra lightglue; pinned source and two local weight files |

Dependency groups are resolved in uv.lock; GPU, browser and device integrations
have not been executed in this cleanup. In particular, AndroidWorld's original
[requirements](https://github.com/google-research/android_world/blob/3e50888527ef9f29b9157ecd537e408008bb1c85/requirements.txt)
pin an older NumPy/Matplotlib environment used with Python 3.11. The current
Python 3.12 group retains AndroidEnv/protobuf/API constraints and resolves newer
NumPy, pandas and Matplotlib. Numerical/device equivalence is pending a restored
emulator validation. Use the frozen tree and original environment record for an
exact historical runtime; do not call the new group a verified reproduction.

After restoring the pinned AndroidWorld source, install its package into this
environment with `uv pip install --no-deps /explicit/path/to/android_world`.
Its setup hook may require the declared grpcio-tools build dependency. Do not
copy site-packages from an old environment. The pinned upstream setup.py is the
source of that build rule. The LightGlue group follows its pinned upstream
[requirements](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/requirements.txt).

Services are started explicitly and use loopback URLs. An actor run requires
`inputs.checkpoint_attestation`, the trusted nonempty checkpoint object printed
by that service, and `services.model`. Guarded replay additionally needs
`services.lightglue`, whose source and weight hashes are checked. Restore models
only after reviewing their own license. No training dependency is required by
this inference/routing project.

Browser and Android configurations in configs/ are starting templates; edit
their task, actor, inputs, service ports and new run ID deliberately. Optional
memory paths point to successful source episodes; parameters.resource_base
relocates relative image references without altering the original receipt bytes.
