# Migration from the restored public tree

Baseline: `8b0eae11468195ce86ccb33fea980334c72ac2d6`.

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
| original batch prompt/decoder | legacy batch runner in frozen archive; decoder in dafc.models.legacy_coordinates |
| c388/c400/c430/c431 pilot helpers | capture/execution/interactions/preconditions libraries |
| c459/c482 Android helpers | elements and device libraries |

Pilot main functions, signed-manifest orchestration, integral repair experiments
and source-bound historical analysis are frozen where current routing has no
dependency on them. Necessary registration predecessors remain implementation
dependencies; they are not alternative public run commands. The retired-symbol
index in `reproduce/frozen/SOURCE.json` records the original scope.

`reproduce/frozen/dafc-8b0eae1.tar.gz` is the entire original public Git tree.
Check its SHA-256 from SOURCE.json before extracting to a new archive directory.
Use that tree for original source-bound verifiers and signed manifests. Do not
change an old hash to admit current source. Every original reproduce/ file and
published statistical input remains unchanged.

New configured runs use `dafc-configured-v1`; new atomic batches have their own
design identity and request IDs and set `paper_evidence: false`. Checkpoint
attestations include weight SHA-256 under `dafc-checkpoint-v2`; regenerate them
from the explicitly selected local model, without editing old research receipts.

Restore relative-path evidence under a new external directory and set
`parameters.resource_base`. Absolute paths inside original evidence are not
rewritten: historical verification uses its original manifest projection; new
runs need a separately reviewed registry with new provenance where paths change.
See TEST_MIGRATION.json for the original test-to-current-test correspondence.
