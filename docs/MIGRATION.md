# Migration

| Previous entry | Maintained entry |
| --- | --- |
| requirements-dev.lock and ad-hoc environments | uv sync --locked; root uv.lock |
| 运行.py and script wrappers | dafc doctor / demo / run --config / reproduce |
| android_atomic_closed_loop.py | dafc run --config with workflow: android-atomic |
| live_miniwob_transition_guard.py | workflow: browser-episode |
| live_androidworld_transition_guard.py | workflow: android-episode |
| src/dafc/transitions.py implementation | transitions API, transition_selection, transition_authorization, risk |
| remote_open_gui_grounder_v3 | dafc.models.grounding |
| remote_open_gui_grounder_v5_service | python -m dafc.models.service |
| remote_open_gui_grounder_v7_qwen_model_frame | dafc.models.model_frame |
| remote_open_gui_grounder_v7_service | python -m dafc.models.model_frame_service |
| original batch prompt/decoder | unused batch runner removed; decoder in dafc.models.legacy_coordinates |
| c388/c400/c430/c431 pilot helpers | capture/execution/interactions/preconditions libraries |
| c459/c482 Android helpers | elements and device libraries |

Pilot-only main functions, unsupported batch runners and unused repair
experiments are removed. Required registration predecessors remain library
dependencies. Four original controller/selection source files are regression
inputs in `tests/reference`; their SHA-256 values are checked before loading.
All scientific reproduction files and published statistical inputs keep their
original bytes.

New configured runs use `dafc-configured-v1`; new atomic batches have their own
design identity and request IDs and set `paper_evidence: false`. Checkpoint
attestations bind the explicitly selected weights under `dafc-checkpoint-v2`.
Do not rewrite old source or evidence hashes. Set `parameters.resource_base` for
explicit external resources; new paths require new experiment provenance.

See [MATERIALS.md](MATERIALS.md) for material ownership and deletion rules.
