"""Run only after portable resource and parameter preflight has succeeded."""

from __future__ import annotations

from typing import Any

from dafc.configuration import RunConfig
from dafc.jobs.atomic import run_batch
from dafc.jobs.platforms import run_episode, service_bindings
from dafc.jobs.preflight import prepare
from dafc.jobs.routing import run_routing
from dafc.run_state import finish, own_run, run_binding


def execute(config: RunConfig) -> dict[str, Any]:
    prepared = prepare(config)
    if prepared["problems"]:
        raise ValueError("; ".join(prepared["problems"]))
    runtime, matcher = service_bindings(config, prepared)
    binding = run_binding(config, runtime)
    with own_run(config, binding) as restored:
        if restored is not None:
            return restored
        if config.workflow == "routing":
            result = run_routing(config)
        elif config.workflow == "android-atomic":
            result = run_batch(config, prepared)
        else:
            result = run_episode(config, prepared, matcher)
        finish(config, binding, result)
        return result
