"""Explicit platform lifecycle. Optional packages load only after preflight succeeds."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from threading import RLock
from typing import Any

from dafc.adapters.android.device import AndroidDevice
from dafc.configuration import RunConfig
from dafc.models import client
from dafc.models.lightglue_runtime_service import LightGlueServiceClient

_ANDROID_BINDING_LOCK = RLock()


def prepare_episode(config: RunConfig) -> dict[str, Any]:
    if config.workflow == "browser-episode":
        from dafc.adapters.browser import observations as support

        task = config.parameters.get("task")
        if task not in support.TASKS:
            raise ValueError(f"Unsupported browser task: {task}")
        seed = config.parameters.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Browser seed must be an integer")
        contract = support.contract_by_name(config.parameters.get("contract", "source"))
        prepared = {"contract": contract}
    else:
        from dafc.adapters.android import observations as support

        task = config.parameters.get("task")
        support.resolved_task_spec(
            task, config.parameters.get("timer_params"), config.parameters.get("browser_task_seed")
        )
        if config.parameters.get("contract", "identity") not in support.CONTRACTS:
            raise ValueError("Unknown Android display contract")
        prepared = {}
    if config.parameters["policy"] != "always":
        if "memory" not in config.inputs:
            raise ValueError("Replay requires inputs.memory with successful source evidence")
        resource_base = (
            config.resources.expand(config.parameters["resource_base"], base=config.config_file.parent)
            if "resource_base" in config.parameters
            else None
        )
        loaded = support.load_memory(config.inputs["memory"].path, resource_base=resource_base)
        prepared.update(memory_goal=loaded[0], memory=loaded[-1])
        if config.workflow == "android-episode":
            prepared["memory_params"] = loaded[1]
    if "risk_certificate" in config.inputs:
        prepared["risk_certificate"] = support.load_risk_certificate(config.inputs["risk_certificate"].path)
    if config.parameters.get("require_risk_certificate") and "risk_certificate" not in prepared:
        raise ValueError("inputs.risk_certificate is required by this run")
    return prepared


def service_bindings(config: RunConfig, prepared: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    if config.workflow == "routing":
        return {}, None
    model = client.health(config.services["model"])
    if model.get("ready") is not True or model.get("kind") != config.parameters["actor"]:
        raise ValueError("Actor service is not ready or its actor differs from configuration")
    if model.get("checkpoint") != prepared["expected_checkpoint"]:
        raise ValueError("Actor checkpoint differs from inputs.checkpoint_attestation")
    receipt = {
        "model": {
            key: model.get(key) for key in ("service_version", "base_runner_version", "kind", "checkpoint")
        }
    }
    matcher = None
    if config.parameters.get("policy") in {"full", "sg", "sgp"}:
        matcher = LightGlueServiceClient(config.services["lightglue"])
        health = matcher.health()
        if health.get("ready") is not True:
            raise ValueError("LightGlue service is not ready")
        from dafc.jobs.service_identity import validate_lightglue

        receipt["lightglue"] = validate_lightglue(health)
    return receipt, matcher.correspondences if matcher is not None else None


@contextmanager
def android_session(config: RunConfig):
    from android_env.wrappers import a11y_grpc_wrapper
    from android_world import registry
    from android_world.env import env_launcher

    device = AndroidDevice(config.inputs["adb"].path, config.parameters.get("serial", "emulator-5554"))
    console_port = config.parameters.get("console_port", 5554)
    if device.serial != f"emulator-{console_port}":
        raise ValueError("serial must match the explicitly selected emulator console_port")
    with _ANDROID_BINDING_LOCK:
        original = a11y_grpc_wrapper._get_accessibility_forwarder_apk
        a11y_grpc_wrapper._get_accessibility_forwarder_apk = config.inputs["a11y_apk"].path.read_bytes
        env = None
        try:
            tasks = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
            env = env_launcher.load_and_setup_env(
                console_port=console_port,
                emulator_setup=False,
                freeze_datetime=True,
                adb_path=str(device.executable),
                grpc_port=config.parameters.get("grpc_port", 8554),
            )
            yield env, tasks, device
        finally:
            a11y_grpc_wrapper._get_accessibility_forwarder_apk = original
            if env is not None:
                try:
                    device.restore_display()
                finally:
                    env.close()


def run_episode(config: RunConfig, prepared: dict[str, Any], matcher: Any) -> dict[str, Any]:
    p = config.parameters
    common = {
        "policy": p["policy"],
        "kind": p["actor"],
        "run_id": config.run_id,
        "match_provider": matcher,
        "require_risk_certificate": p.get("require_risk_certificate", False),
        "score_postcondition_for_ablations": p.get("score_postcondition_for_ablations", False),
    }
    for key in ("memory", "memory_goal", "memory_params", "risk_certificate"):
        if key in prepared:
            common[key] = prepared[key]
    if config.workflow == "browser-episode":
        from miniwob.environment import MiniWoBEnvironment

        from dafc.adapters.browser.episode import run_episode as execute
        from dafc.adapters.browser.miniwob_agent_harness import EXPECTED_CHROMEDRIVER_SHA256, driver_binding
        from dafc.adapters.browser.observations import service_grounder

        def create_environment(**kwargs):
            with driver_binding(config.inputs["chromedriver"].path, EXPECTED_CHROMEDRIVER_SHA256):
                return MiniWoBEnvironment(**kwargs)

        return execute(
            **common,
            task=p["task"],
            seed=p["seed"],
            contract=prepared["contract"],
            output_dir=config.output,
            grounder=partial(service_grounder, config.services["model"], p["actor"]),
            environment_factory=create_environment,
            geometry_frontend="lightglue" if matcher else "classical_sift",
            memory_enable_after_steps=p.get("memory_enable_after_steps", 0),
        )
    from dafc.adapters.android.episode import run_episode as execute

    with android_session(config) as (env, tasks, device):
        return execute(
            **common,
            env=env,
            task_registry=tasks,
            device=device,
            task_id=p["task"],
            contract_id=p.get("contract", "identity"),
            model_service_url=config.services["model"],
            output_root=config.output,
            timer_params=p.get("timer_params"),
            browser_task_seed=p.get("browser_task_seed"),
        )
