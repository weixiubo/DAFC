import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dafc import routing
from dafc.evidence import AppendOnlyLedger, atomic_write_json, sha256
from dafc.registration import action_contract


def assert_receipt_matches(actual, expected, path="receipt"):
    """Preserve receipt decisions and counts while allowing float roundoff."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            assert_receipt_matches(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            assert_receipt_matches(value, reference, f"{path}[{index}]")
    elif isinstance(expected, float):
        # OpenCV's numerical kernels can differ in their last bits across CPUs.
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), path
    else:
        assert actual == expected, path


def test_original_seeded_geometry_receipts():
    expected = json.loads((Path(__file__).parent / "fixtures/core-baseline.json").read_text())
    rng = np.random.default_rng(585001)
    src = cv2.GaussianBlur(rng.integers(0, 256, size=(360, 480, 3), dtype=np.uint8), (3, 3), 0)
    targets = [
        src.copy(),
        cv2.warpAffine(src, np.float32([[1, 0, 12], [0, 1, 8]]), (480, 360)),
        np.zeros_like(src),
    ]
    for dst, wanted in zip(targets, expected, strict=True):
        receipt = routing.standard_registration(src, dst, [211.0, 143.0])
        receipt.pop("runtime_s", None)
        assert_receipt_matches(receipt, wanted["standard"])
        assert_receipt_matches(
            action_contract.register_action_contract(src, dst, [211.0, 143.0]), wanted["dafc"]
        )


def test_receipt_comparison_allows_machine_roundoff():
    assert_receipt_matches({"matrix": [[1.0 + 2e-16, 3e-17]]}, {"matrix": [[1.0, 0.0]]})


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"mapped_action_xy": [211.001, 143.0]}, {"mapped_action_xy": [211.0, 143.0]}),
        ({"accepted": False}, {"accepted": True}),
        ({"inliers": 4500}, {"inliers": 4501}),
        ({"reason": "rejected"}, {"reason": "accepted"}),
    ],
)
def test_receipt_comparison_rejects_behavior_changes(actual, expected):
    with pytest.raises(AssertionError):
        assert_receipt_matches(actual, expected)


def test_source_tampering_fails_closed(tmp_path):
    image = tmp_path / "source.png"
    cv2.imwrite(str(image), np.zeros((20, 20, 3), dtype=np.uint8))
    source = {
        "source_success": True,
        "atomic_memory": {
            "source_before": str(image),
            "source_before_sha256": "0" * 64,
            "source_action_xy": [10, 10],
        },
    }
    with pytest.raises(RuntimeError, match="changed"):
        routing.authorize("dafc", source, np.zeros((20, 20, 3), dtype=np.uint8))


def test_ledger_integrity_and_no_overwrite(tmp_path):
    artifact = tmp_path / "episode.json"
    atomic_write_json(artifact, {"success": False})
    with pytest.raises(FileExistsError):
        atomic_write_json(artifact, {"success": True})
    ledger = AppendOnlyLedger(tmp_path / "ledger.jsonl", tmp_path)
    row = {
        "record_id": "r1",
        "artifact": "episode.json",
        "artifact_sha256": sha256(artifact),
        "status": "failed",
    }
    ledger.append(row)
    with pytest.raises(FileExistsError):
        ledger.append(row)
    artifact.write_text("{}")
    with pytest.raises(RuntimeError, match="mismatch"):
        ledger.verify()
    with pytest.raises(RuntimeError, match="escapes"):
        ledger.artifact_path("../outside")
