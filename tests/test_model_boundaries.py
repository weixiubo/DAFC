"""Service and matcher identities are validated independently of GPU inference."""

import base64

import pytest

from dafc.jobs.service_identity import validate_lightglue
from dafc.models import grounding, model_frame, service
from dafc.models.lightglue_runtime_frontend import (
    EXPECTED_REPO_COMMIT,
    EXPECTED_WEIGHT_HASHES,
    FRONTEND_VERSION,
)
from dafc.models.lightglue_runtime_service import SERVICE_VERSION


def test_checkpoint_detects_same_size_weight_changes(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"first")
    first = grounding.checkpoint_fingerprint(tmp_path)
    weight.write_bytes(b"other")
    second = grounding.checkpoint_fingerprint(tmp_path)
    assert first["total_weight_bytes"] == second["total_weight_bytes"]
    assert first["weight_files"][0]["sha256"] != second["weight_files"][0]["sha256"]


def test_model_frame_passes_callbacks_without_module_mutation(monkeypatch):
    original_prompt, original_parser = grounding.prompt, grounding.parse_normalized_click

    def infer(*args, **kwargs):
        assert kwargs == {
            "prompt_builder": model_frame.qwen_model_frame_prompt,
            "action_parser": model_frame.parse_model_frame_click,
        }
        assert grounding.prompt is original_prompt
        assert grounding.parse_normalized_click is original_parser
        return {"image_width": 320, "image_height": 240}

    monkeypatch.setattr(grounding, "infer_one", infer)
    result = model_frame.infer_one(None, None, "qwen25vl", {})
    assert result["coordinate_interface"]["executor_maps_to_original_screenshot"] is True
    with pytest.raises(ValueError):
        model_frame.load_model("uitars", None)


def test_servers_keep_their_own_backend_identity(tmp_path):
    first = service.GrounderServer(
        ("127.0.0.1", 0), None, None, "uitars", tmp_path / "one.jsonl", {}, runner=grounding
    )
    second = service.GrounderServer(
        ("127.0.0.1", 0),
        None,
        None,
        "qwen25vl",
        tmp_path / "two.jsonl",
        {},
        runner=model_frame,
        service_version="model-frame",
    )
    try:
        assert first.runner is grounding
        assert second.runner is model_frame
        assert first.service_version != second.service_version
    finally:
        first.server_close()
        second.server_close()


@pytest.mark.parametrize("damage", [None, "revision", "weights", "version"])
def test_lightglue_health_binds_source_and_weights(damage):
    receipt = {
        "service_version": SERVICE_VERSION,
        "bindings": {
            "frontend_version": FRONTEND_VERSION,
            "repo": {"commit": EXPECTED_REPO_COMMIT},
            "weights": {key: {"sha256": value} for key, value in EXPECTED_WEIGHT_HASHES.items()},
        },
    }
    if damage == "revision":
        receipt["bindings"]["repo"]["commit"] = "wrong"
    if damage == "weights":
        receipt["bindings"]["weights"].clear()
    if damage == "version":
        receipt["service_version"] = "wrong"
    if damage is None:
        assert validate_lightglue(receipt) == receipt
    else:
        with pytest.raises(ValueError):
            validate_lightglue(receipt)


def test_service_rejects_wrong_image_hash_and_duplicate_audit_ids(tmp_path):
    payload = {
        "id": "x",
        "image_base64": base64.b64encode(b"image").decode(),
        "image_sha256": "0" * 64,
        "instruction": "click",
    }
    with pytest.raises(ValueError, match="image_sha256"):
        service.decode_request(payload, "qwen25vl")
    audit = tmp_path / "audit.jsonl"
    service.durable_append(audit, {"request": {"id": "x"}})
    service.durable_append(audit, {"request": {"id": "x"}})
    with pytest.raises(RuntimeError, match="duplicate"):
        service.existing_ids(audit)
