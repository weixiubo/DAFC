# Resources and configuration

Defaults are `~/research/data/DAFC`, `~/research/models`,
`~/research/runs/DAFC` and `~/research/data/DAFC/evidence`. Resolution precedence
is CLI options, explicit YAML paths, environment variables, user JSON, then
defaults. Environment variables are `DAFC_DATA_DIR`, `DAFC_MODELS_DIR`,
`DAFC_RUNS_DIR` and `DAFC_EVIDENCE_DIR`.

User configuration is `$XDG_CONFIG_HOME/research/DAFC.json`, normally
`~/.config/research/DAFC.json`; `DAFC_LOCAL_CONFIG` selects another file.

```json
{
  "paths": {
    "data": "/path/to/data",
    "models": "/path/to/models",
    "runs": "/path/to/runs",
    "evidence": "/path/to/data/DAFC/evidence"
  }
}
```

Individual catalog resources can be set in a `resources` map or through
`DAFC_RESOURCE_<ID>`, using uppercase letters and underscores. YAML paths
accept `{data}`, `{models}`, `{runs}` and `{evidence}`; relative values resolve
from the configuration file.

`doctor --require ID` checks a required catalog resource.
`doctor --config FILE` checks input hashes, workflow prerequisites and optional
dependencies. The [catalog](../src/dafc/resource_catalog.json) records source,
revision, license references, locations and acquisition instructions. Public
statistical inputs are packaged; original images are an additional input to
`dafc reproduce --raw-root PATH`.

## Optional environments

| Workflow | Setup |
| --- | --- |
| Routing, tests and public evidence | `uv sync --locked` |
| Browser episodes | `--extra browser`, matching Chrome/driver and a task server |
| Android episodes and batches | `--extra android`, pinned AndroidWorld source, emulator, SDK and APK |
| Model service | `--extra inference` and a local checkpoint |
| Learned matcher | `--extra lightglue`, pinned source and two local weight files |

All extras are resolved in `uv.lock`. The original AndroidWorld revision's
[requirements](https://github.com/google-research/android_world/blob/3e50888527ef9f29b9157ecd537e408008bb1c85/requirements.txt)
describe its Python 3.11 experiment environment. The current Python 3.12
integration retains AndroidEnv/protobuf/API constraints and uses newer NumPy,
pandas and Matplotlib versions. Exact original-runtime execution uses its
recorded dependency set; the current integration uses the root lock.

Install the pinned AndroidWorld source with
`uv pip install --no-deps /path/to/android_world` after selecting the Android
extra. Its setup uses the declared `grpcio-tools` dependency. The LightGlue
group follows its pinned upstream
[requirements](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/requirements.txt).

## Service and episode configuration

Services use explicitly configured loopback URLs. Actor runs require
`services.model` and `inputs.checkpoint_attestation`, using the checkpoint
identity returned by that service. Guarded replay also uses
`services.lightglue`, with source and weight hashes checked before execution.

Copy a browser or Android template from `configs/` and set the task, actor,
inputs, service ports and new run ID. Memory paths identify successful source
episodes. `parameters.resource_base` relocates relative image references
while preserving receipt bytes. Model and dataset terms are summarized in
[third-party notices](../THIRD_PARTY.md).
