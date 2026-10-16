"""Inspect distributions and run installed functionality outside the source tree."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def inspect_distributions():
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    sources = sorted((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Build one current wheel and sdist into a clean dist directory first")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        assert any(name.endswith("/licenses/LICENSE") for name in names)
        metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
        assert "License-Expression: Apache-2.0" in archive.read(metadata).decode()
        for name in (
            "resource_catalog.json",
            "evidence_data/rows.json",
            "evidence_data/expected.json",
            "evidence_data/checksums.json",
            "evidence_data/frozen_statistics.py",
        ):
            assert "dafc/" + name in names, name
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        for suffix in (
            "/LICENSE",
            "/uv.lock",
            "/tests/reference/MANIFEST.json",
            "/tests/test_atomic_phase_equivalence.py",
        ):
            assert any(name.endswith(suffix) for name in names), suffix
    return wheels[0]


def _command(args, *, cwd, env):
    completed = subprocess.run(list(map(str, args)), cwd=cwd, env=env, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    return completed.stdout


def check_install(wheel, temporary):
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    environment = temporary / "environment"
    install_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT", "UV_PROJECT_ENVIRONMENT"}
    }
    install_env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    # A locked sync needs cached wheels, not a previously queried package index.
    _command(
        [uv, "sync", "--locked", "--offline", "--no-dev", "--no-install-project", "--python", sys.executable],
        cwd=ROOT,
        env=install_env,
    )
    python = environment / "bin/python"
    _command(
        [uv, "pip", "install", "--offline", "--no-deps", "--python", python, wheel],
        cwd=temporary,
        env=None,
    )
    _command([uv, "pip", "check", "--python", python], cwd=temporary, env=None)
    outside = temporary / "unrelated"
    outside.mkdir()
    env = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH" and not key.startswith("DAFC_")
    }
    env.update(
        HOME=str(temporary / "home"),
        XDG_CONFIG_HOME=str(temporary / "home/.config"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    executable = environment / "bin/dafc"
    results = {
        command: json.loads(_command([executable, command], cwd=outside, env=env))
        for command in ("doctor", "demo", "reproduce")
    }
    assert results["doctor"]["status"] == "ok"
    assert results["demo"]["reuse_authorized"] and results["demo"]["failed_source_falls_back"]
    assert results["reproduce"]["outcomes"] == 1120
    assert results["reproduce"]["historical_result"] is True
    _command(
        [
            python,
            "-c",
            "import importlib,pkgutil,dafc; [importlib.import_module(item.name) for item in pkgutil.walk_packages(dafc.__path__, dafc.__name__+'.')]",
        ],
        cwd=outside,
        env=env,
    )
    assert not list(outside.iterdir()), "Import or basic commands wrote into caller CWD"
    assert not (temporary / "home").exists(), "Import or diagnostics created user resource directories"
    inputs = temporary / "inputs"
    _command([python, ROOT / "examples/make_routing_inputs.py", "--directory", inputs], cwd=outside, env=env)
    configuration = temporary / "routing.json"
    configuration.write_text(
        json.dumps(
            {
                "workflow": "routing",
                "run_id": "wheel-run",
                "output": str(temporary / "result"),
                "inputs": {
                    "source": {"path": str(inputs / "source.png")},
                    "target": {"path": str(inputs / "target.png")},
                },
                "parameters": {"source_success": True, "action_xy": [160, 120]},
            }
        )
    )
    result = json.loads(_command([executable, "run", "--config", configuration], cwd=outside, env=env))
    assert result["decision"]["accepted"] is True
    return {
        "doctor": "ok",
        "demo": "geometric transfer verified",
        "run": "installed routing accepted",
        "reproduce": "1120 outcomes verified",
        "imports": "all maintained modules; no optional dependencies",
    }


def main():
    wheel = inspect_distributions()
    with TemporaryDirectory(prefix="dafc-release-") as directory:
        checks = check_install(wheel, Path(directory))
    print(json.dumps({"status": "ok", "wheel": wheel.name, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
