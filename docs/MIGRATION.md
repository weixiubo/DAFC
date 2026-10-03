# Migration

| Previous entry | Current entry |
| --- | --- |
| `requirements-dev.lock` and separate environment snapshots | `uv sync --locked` and root `uv.lock` |
| `运行.py` and script wrappers | `dafc doctor`, `demo`, `run --config`, `reproduce` |
| `android_atomic_closed_loop.py` | `run --config` with `workflow: android-atomic` |
| `live_miniwob_transition_guard.py` | `workflow: browser-episode` |
| `live_androidworld_transition_guard.py` | `workflow: android-episode` |
| `src/dafc/transitions.py` implementation | `transitions` API, `transition_selection`, `transition_authorization`, `risk` |
| `remote_open_gui_grounder_v3` | `dafc.models.grounding` |
| `remote_open_gui_grounder_v5_service` | `python -m dafc.models.service` |
| `remote_open_gui_grounder_v7_qwen_model_frame` | `dafc.models.model_frame` |
| `remote_open_gui_grounder_v7_service` | `python -m dafc.models.model_frame_service` |
| Batch coordinate decoder | `dafc.models.legacy_coordinates` |
| c388/c400/c430/c431 helpers | Capture, execution, interaction and precondition libraries |
| c459/c482 Android helpers | Element and device libraries |

Configured runs use the `dafc-configured-v1` identity. Atomic batches record
their design and request IDs. Checkpoint receipts use `dafc-checkpoint-v2`
to bind the selected model files. `parameters.resource_base` resolves relative
image references within explicitly supplied evidence.

Regression comparisons use four original controller/selection files identified
in `tests/reference/MANIFEST.json`. Scientific inputs retain their producer
identities; each current experiment records its own source and resource bindings.
