"""Deterministic platform stand-ins for the original and current episode controllers."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def frozen_controller(platform, support):
    metadata = json.loads((ROOT / "reproduce/frozen/SOURCE.json").read_text())
    archive = ROOT / "reproduce/frozen" / metadata["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
    name = "live_miniwob_transition_guard" if platform == "browser" else "live_androidworld_transition_guard"
    with tarfile.open(archive) as bundle:
        source = bundle.extractfile(f"src/dafc/adapters/{platform}/{name}.py").read().decode()
    fn = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_episode"
    )
    # Only the optional import statements are replaced by the platform stand-ins below.
    fn.body = [node for node in fn.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ast.parse("from __future__ import annotations")
    module.body.append(fn)
    namespace = dict(vars(support))
    namespace["MiniWoBEnvironment"] = FakeBrowser
    exec(compile(ast.fix_missing_locations(module), "frozen-original-controller", "exec"), namespace)
    return namespace["run_episode"], namespace


class FakeDriver:
    def set_window_size(self, *args):
        pass

    def execute_script(self, script):
        return 0


class FakeBrowser:
    def __init__(self, **kwargs):
        self.instance = SimpleNamespace(driver=FakeDriver())
        self.actions = 0
        self.closed = False

    def close(self):
        self.closed = True


class FakeAndroid:
    def __init__(self):
        self.actions = 0

    def reset(self, **kwargs):
        pass

    def get_state(self, **kwargs):
        return SimpleNamespace(pixels=np.zeros((80, 100, 3), dtype=np.uint8))


class Clock:
    def __init__(self):
        self.ticks = 0

    def perf_counter(self):
        self.ticks += 1
        return self.ticks / 10


def normalized(value, directory):
    if isinstance(value, dict):
        return {key: normalized(item, directory) for key, item in value.items() if key != "elapsed_s"}
    if isinstance(value, list):
        return [normalized(item, directory) for item in value]
    if isinstance(value, str):
        return value.replace(str(directory), "<RUN>")
    return value


def exercise(platform, policy, scenario, directory, monkeypatch, *, original):
    support = importlib.import_module(f"dafc.adapters.{platform}.observations")
    phases = importlib.import_module(f"dafc.adapters.{platform}.episode_steps")
    report = importlib.import_module(f"dafc.adapters.{platform}.episode_report")
    entry = importlib.import_module(f"dafc.adapters.{platform}.episode")
    old, namespace = frozen_controller(platform, support)
    namespace.update(vars(phases))
    namespace.update(vars(entry))
    events = []
    clock = Clock()
    for module in (phases, report, entry):
        if hasattr(module, "time"):
            monkeypatch.setattr(module, "time", clock)
    namespace["time"] = clock
    env = FakeBrowser() if platform == "browser" else FakeAndroid()
    if platform == "browser":
        namespace["MiniWoBEnvironment"] = lambda **kwargs: env
    memory = [
        SimpleNamespace(
            transition_id="m0",
            demonstration_order=0,
            source_success=True,
            action_xy=(20.0, 30.0),
            before=np.zeros((80, 100, 3), dtype=np.uint8),
            terminal=True,
        )
    ]

    def select(*args, **kwargs):
        events.append("select")
        decision = {
            "decision": "reuse",
            "selected_transition_id": "m0",
            "progress_recovered_transition_ids": [],
        }
        return decision, SimpleNamespace(transition_id="m0", mapped_action_xy=(20.0, 30.0))

    def verify(*args):
        events.append("verify-post")
        return {"post_verified": scenario != "post-failure"}

    guard = SimpleNamespace(METHOD_VERSION=support.GUARD.METHOD_VERSION, select=select, verify_post=verify)
    monkeypatch.setattr(phases, "GUARD", guard)
    monkeypatch.setattr(report, "GUARD", guard)
    namespace["GUARD"] = guard

    def infer(*args, **kwargs):
        events.append("model")
        if scenario == "exception":
            raise RuntimeError("model unavailable")
        return {
            "result": {"parsed_action": {"valid": scenario != "parse-failure", "normalized_xy": [0.25, 0.4]}}
        }

    def capture(path, image=None, contract=None):
        if platform == "browser":
            path = image
        events.append("capture")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.full((80, 100, 3), env.actions, dtype=np.uint8)
        assert cv2.imwrite(str(path), data)
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": 100,
            "height": 80,
            **({"format": "png"} if platform == "browser" else {}),
        }

    def terminal():
        return scenario != "exhausted" and env.actions >= (2 if scenario == "post-failure" else 1)

    def browser_click(active, point):
        events.append("execute")
        env.actions += 1
        return None, {
            "reward": float(terminal()),
            "raw_reward": float(terminal()),
            "terminated": terminal(),
            "truncated": False,
            "done": terminal(),
        }

    def android_click(active, point, image):
        events.append("execute")
        env.actions += 1
        return tuple(float(round(value)) for value in point)

    kwargs = dict(
        policy=policy,
        kind="qwen25vl",
        run_id="regression",
        attempt=0,
        memory=memory if policy != "always" else None,
        memory_goal="goal" if policy != "always" else None,
        match_provider=(lambda *args: {}) if policy in {"full", "sg", "sgp"} else None,
        score_postcondition_for_ablations=policy in {"sg", "sgp"},
    )
    if platform == "browser":
        monkeypatch.setattr(phases.C388, "capture", capture)
        monkeypatch.setattr(phases.C400, "reset_harness", lambda *args: events.append("reset-harness"))
        monkeypatch.setattr(phases.C430, "click", browser_click)
        monkeypatch.setattr(phases.C430, "wait_for_visual_settle", lambda *args: None)
        monkeypatch.setattr(phases.HARNESS, "reset_agent_episode", lambda *args: ({"utterance": "goal"}, {}))
        kwargs.update(
            task="click-option",
            seed=124,
            contract={"name": "source", "format": "png", "supported": True, "capture_scale": 1.0},
            output_dir=directory,
            grounder=infer,
        )
        if not original:
            kwargs["environment_factory"] = lambda **kwargs: env
    else:

        class Task:
            goal = "goal"

            def __init__(self, params):
                pass

            def initialize_task(self, env):
                events.append("initialize")

            def is_successful(self, env):
                events.append("score")
                return float(terminal())

            def tear_down(self, env):
                events.append("tear-down")

        fake_device = SimpleNamespace(apply_contract=lambda *args: events.append("contract"))
        namespace["C482"] = SimpleNamespace(**vars(phases.C482), apply_contract=fake_device.apply_contract)
        if not original:
            kwargs["device"] = fake_device
        monkeypatch.setattr(phases.C482, "pixels", lambda state: state.pixels)
        monkeypatch.setattr(phases.C482, "observe", lambda image, contract: image)
        replacements = {
            "capture": capture,
            "execute_click": android_click,
            "open_app": lambda *args: events.append("open-app"),
            "ensure_clean_observation_ui": lambda env, stage, **kwargs: {
                "stage": stage,
                "pointer_location_after_cleanup": "0",
            },
        }
        for name, value in replacements.items():
            monkeypatch.setattr(phases, name, value)
            namespace[name] = value
        monkeypatch.setattr(phases.GROUND_CLIENT, "infer", infer)
        kwargs.update(
            env=env,
            task_registry={"ClockStopWatchRunning": Task},
            task_id="ClockStopWatchRunning",
            contract_id="identity",
            model_service_url="http://127.0.0.1:1",
            output_root=directory,
            memory_params={} if policy != "always" else None,
        )
    try:
        result = (old if original else entry.run_episode)(**kwargs)
    except RuntimeError as exc:
        result = {"exception": str(exc)}
    if platform == "browser":
        events.append("closed" if env.closed else "leaked")
    return {"result": normalized(result, directory), "events": events, "actions": env.actions}
