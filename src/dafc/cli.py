"""Installed DAFC commands; read-only diagnostics and explicit configured runs."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from dafc.configuration import load_config
from dafc.resources import resolve_paths, resource_report


def _paths(parser: argparse.ArgumentParser) -> None:
    for name in ("data", "models", "runs", "evidence"):
        parser.add_argument(f"--{name}-dir", type=Path)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Read-only environment and resource checks")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--require", action="append", default=[], help="Require a catalog resource ID")
    _paths(doctor)
    commands.add_parser("demo", help="Small deterministic synthetic routing example")
    run = commands.add_parser("run", help="Execute an explicit YAML configuration")
    run.add_argument("--config", type=Path, required=True)
    _paths(run)
    reproduce = commands.add_parser("reproduce", help="Recompute frozen public estimates")
    reproduce.add_argument("--raw-root", type=Path)
    return root


def doctor(args: argparse.Namespace) -> dict[str, Any]:
    from dafc.jobs.preflight import prepare

    overrides = {key: getattr(args, key + "_dir") for key in ("data", "models", "runs", "evidence")}
    cfg = load_config(args.config, overrides) if args.config else None
    paths = cfg.resources if cfg else resolve_paths(overrides)
    catalog = json.loads((Path(__file__).parent / "resource_catalog.json").read_text())
    report = {
        "status": "ok",
        "python": platform.python_version(),
        "paths": paths.as_strings(),
        "resources": resource_report(catalog, paths),
        "problems": prepare(cfg)["problems"] if cfg else [],
    }
    indexed = {row["id"]: row for row in report["resources"]}
    for name in args.require:
        if name not in indexed:
            raise ValueError(f"Unknown resource ID {name}")
        item = indexed[name]
        if item["status"] not in {"ready", "present_unverified"}:
            report["problems"].append(f"{name}: {item['status']} at {item['path']}; {item['how_to_get']}")
    if report["problems"]:
        report["status"] = "missing_requirements"
    return report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args)
            code = 3 if result["problems"] else 0
        elif args.command == "demo":
            from dafc.demo import demo

            result, code = demo(), 0
        elif args.command == "reproduce":
            from dafc.reproduction import verify

            result, code = verify(args.raw_root), 0
        else:
            from dafc.jobs.run import execute

            overrides = {key: getattr(args, key + "_dir") for key in ("data", "models", "runs", "evidence")}
            result = execute(load_config(args.config, overrides))
            code = 4 if result.get("status") == "incomplete" else 0
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return code
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
