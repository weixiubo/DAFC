from __future__ import annotations

import pytest

from .episode_harness import exercise


@pytest.mark.parametrize("platform", ["browser", "android"])
@pytest.mark.parametrize(
    "policy,scenario",
    [
        ("always", "success"),
        ("always", "parse-failure"),
        ("always", "exception"),
        ("always", "exhausted"),
        ("full", "success"),
        ("full", "post-failure"),
        ("sg", "success"),
        ("sgp", "post-failure"),
        ("raw", "success"),
        ("naive", "success"),
    ],
)
def test_episode_events_and_outcomes_match_original(platform, policy, scenario, tmp_path, monkeypatch):
    with monkeypatch.context() as context:
        expected = exercise(platform, policy, scenario, tmp_path / "original", context, original=True)
    with monkeypatch.context() as context:
        actual = exercise(platform, policy, scenario, tmp_path / "current", context, original=False)
    assert actual == expected
    if platform == "browser":
        assert actual["events"][-1] == "closed"
    else:
        assert actual["events"][-1] == "tear-down"
    if scenario == "parse-failure":
        assert actual["actions"] == 0
        assert actual["result"]["model_calls"] == 1
