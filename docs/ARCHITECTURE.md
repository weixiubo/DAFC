# Architecture

`routing` implements the action-level S/G decision and fallback contract. The
registration stack shares the required fitting, selection and action-certificate
algorithms; public callers select a policy, not an implementation version.
`coordinates` normalizes typed coordinate payloads independently of experiments.

`transition_authorization` evaluates geometric/precondition evidence.
`transition_selection.SelectionState` owns confirmation, recovered progress and
observable-prefix state; selection preserves demonstration order and fail-closed
terminal gating. Matchers are explicit arguments, never global monkey patches.
`preconditions` and `risk` contain platform-independent scientific rules.

Each platform has immutable EpisodeConfig, EpisodeState and StepState objects.
The controller sequences validation, observation, selection, action resolution,
execution, confirmation and receipt recording. Platform observations and device
operations have separate modules. AndroidDevice binds the executable and serial
explicitly. Upstream browser/Android hook substitutions are scoped, locked and
restored after use; importing the package does not bind a device or driver.

Atomic trials use TrialConfig/TrialState and separate setup, action decision,
execution, outcome and cleanup phases. Attempt storage retains the original
one-predecision-retry rule. A ledger durably binds each attempt hash. Recovery
verifies the experiment design, screenshots and ledger before another action.
An in-flight action without a durable outcome requires manual inspection; the
runner cannot infer whether an emulator already executed it.

`configuration`, `jobs.preflight` and `resources` validate explicit inputs before
output creation. `run_state` binds code, inputs, parameters and service identities,
and owns an exclusive run lock. Completed results have checksums; non-atomic live
episodes cannot be resumed by replaying actions. Atomic batches may resume their
remaining permitted jobs only when all bindings and lifecycle records agree.

Model prompts, decoding and the model-frame strategy are explicit functions.
Service instances own their backend, so one model-frame service cannot change a
different service through imported globals. Optional frameworks load at use time.
The checkpoint receipt hashes model file contents as well as configuration.
