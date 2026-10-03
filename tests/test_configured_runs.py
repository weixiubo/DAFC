"""Public configuration, installed behavior and fail-closed recovery contracts."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from dafc.cli import main
from dafc.configuration import load_config
from dafc.jobs.run import execute
from dafc.run_state import check_resume, finish, own_run, run_binding, save_status


@pytest.fixture
def configured(tmp_path):
    image = np.random.default_rng(585001).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "source.png"), image)
    cv2.imwrite(str(tmp_path / "target.png"), image)
    payload = {
        "workflow": "routing",
        "run_id": "offline",
        "output": "result",
        "inputs": {"source": {"path": "source.png"}, "target": {"path": "target.png"}},
        "parameters": {"action_xy": [160, 120], "source_success": True},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path, payload


def test_routing_and_completed_resume_bind_results(configured, monkeypatch, tmp_path):
    path, _ = configured
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)
    cfg = load_config(path)
    result = execute(cfg)
    assert result["decision"]["accepted"] is True
    assert result["decision"]["mapped_action_xy"] == pytest.approx([160, 120], abs=0.05)
    assert list(outside.iterdir()) == []
    resumed = replace(cfg, resume=True)
    assert execute(resumed) == result
    with pytest.raises(FileExistsError, match="already exists"):
        execute(cfg)
    cfg.output.joinpath("result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        execute(resumed)


def test_resume_rejects_changed_input_and_configuration(configured):
    path, _ = configured
    cfg = load_config(path)
    execute(cfg)
    with pytest.raises(ValueError, match="mismatch"):
        execute(replace(cfg, resume=True, parameters={**cfg.parameters, "source_success": False}))
    cv2.imwrite(str(cfg.inputs["target"].path), np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="mismatch"):
        execute(replace(cfg, resume=True))


@pytest.mark.parametrize(
    "change",
    [
        {"workflow": []},
        {"paths": {"data": 17}},
        {"output": []},
        {"run_id": "../escape"},
        {"inputs": {"source": {"path": "source.png", "sha256": "bad"}}},
        {"services": {"model": "ftp://localhost"}},
        {"resume": "yes"},
        {"surprise": True},
    ],
)
def test_malformed_configuration_fails_before_output(configured, change):
    path, payload = configured
    path.write_text(yaml.safe_dump({**payload, **change}))
    with pytest.raises(ValueError):
        load_config(path)
    assert not path.with_name("result").exists()


@pytest.mark.parametrize(
    "parameters",
    [
        {"source_success": "yes", "action_xy": [1, 2]},
        {"source_success": True, "action_xy": [True, 2]},
        {"source_success": True, "action_xy": [float("nan"), 2]},
        {"source_success": True, "action_xy": [1, 2], "policy": []},
    ],
)
def test_bad_parameters_are_preflight_errors(configured, parameters):
    path, payload = configured
    path.write_text(yaml.safe_dump({**payload, "parameters": parameters}))
    with pytest.raises(ValueError):
        execute(load_config(path))
    assert not path.with_name("result").exists()


def test_missing_and_corrupt_resources_are_read_only(configured, capsys):
    path, payload = configured
    payload["inputs"]["source"]["sha256"] = "0" * 64
    payload["inputs"]["target"]["path"] = "absent.png"
    path.write_text(yaml.safe_dump(payload))
    assert main(["doctor", "--config", str(path)]) == 3
    report = json.loads(capsys.readouterr().out)
    assert any("checksum mismatch" in problem for problem in report["problems"])
    assert any("missing" in problem for problem in report["problems"])
    assert main(["run", "--config", str(path)]) == 2
    assert not path.with_name("result").exists()


def test_configuration_paths_override_environment_and_command_wins(configured, monkeypatch, tmp_path):
    path, payload = configured
    monkeypatch.setenv("DAFC_DATA_DIR", str(tmp_path / "environment"))
    payload["paths"] = {"data": "configured"}
    path.write_text(yaml.safe_dump(payload))
    assert load_config(path).resources.data == tmp_path / "configured"
    assert load_config(path, {"data": tmp_path / "command"}).resources.data == tmp_path / "command"


def test_interrupted_episode_refuses_replay_and_exclusive_owner(configured):
    cfg = load_config(configured[0])
    binding = run_binding(cfg, {})
    with pytest.raises(RuntimeError, match="interrupted"):
        with own_run(cfg, binding):
            with pytest.raises(BlockingIOError):
                with own_run(replace(cfg, workflow="android-atomic", resume=True), binding):
                    pytest.fail("a second owner entered")
            raise RuntimeError("interrupted")
    with pytest.raises(ValueError, match="cannot be safely replayed"):
        check_resume(replace(cfg, resume=True), binding)


def test_atomic_interruption_can_resume_with_identical_binding(configured):
    cfg = replace(load_config(configured[0]), workflow="android-atomic", resume=True)
    cfg.output.mkdir()
    binding = run_binding(cfg, {})
    save_status(cfg.output, binding, "interrupted")
    with own_run(cfg, binding) as previous:
        assert previous is None
        finish(cfg, binding, {"outcomes": 1})
    assert check_resume(cfg, binding) == {"outcomes": 1}


def test_model_loader_imports_torch_and_uses_only_local_files(monkeypatch, tmp_path):
    from dafc.models.grounding import load_model

    calls = []
    model = SimpleNamespace(eval=lambda: calls.append("eval"))

    def loader(path, **kwargs):
        calls.append(kwargs)
        return model

    factory = SimpleNamespace(from_pretrained=loader)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=factory,
            Qwen2_5_VLForConditionalGeneration=factory,
            Qwen2VLForConditionalGeneration=factory,
        ),
    )
    assert load_model("qwen25vl", tmp_path) == (model, model)
    assert all(row["local_files_only"] for row in calls if isinstance(row, dict))
    assert calls[1]["torch_dtype"] == "bf16"
    assert calls[-1] == "eval"
