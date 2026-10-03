# Architecture

`routing` implements the action-level source/geometry decision and fallback
contract. Registration components provide fitting, selection and action
certificates. Callers select a policy; `coordinates` normalizes typed
coordinate payloads independently of experiment execution.

`transition_authorization` evaluates geometry and precondition evidence.
`transition_selection.SelectionState` tracks confirmation, recovered progress
and observable-prefix state. Selection preserves demonstration order and
requires terminal authorization. Matchers are explicit parameters.
`preconditions` and `risk` hold platform-independent rules.

## Platform episodes

`EpisodeConfig`, `EpisodeState` and `StepState` separate configuration and
execution state. Controllers sequence validation, observation, selection,
action resolution, execution, confirmation and receipt recording. Observation
adapters and device operations have separate modules. `AndroidDevice` binds
its executable and serial explicitly. Upstream hook substitutions are scoped,
synchronized and restored after use.

## Atomic trials and recovery

`TrialConfig` and `TrialState` support setup, action decisions, execution,
outcome handling and cleanup. Attempt storage applies the protocol's single
predecision retry. A durable ledger binds each attempt hash. Recovery checks
design identity, screenshots and ledger records before another action.
An in-flight action without a durable outcome requires inspection of the
device state before continuing.

`configuration`, `jobs.preflight` and `resources` validate inputs before
output creation. `run_state` binds code, input, parameter and service identities
and owns the run lock. Completed results carry checksums. Atomic batches
resume remaining permitted jobs when lifecycle records agree; other live
episodes use a new run rather than replaying completed actions.

## Model services

Prompts, decoding and model-frame strategies have explicit interfaces.
Each service owns its backend. Optional frameworks load at use time, and
checkpoint receipts bind model-file contents and configuration.
