#!/usr/bin/env python3
"""Resumable, append-only MiniWoB formal experiment for TransitionGuard."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import json
import subprocess
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import dafc.evidence as formal
import dafc.adapters.browser.live_miniwob_transition_guard as live
from dafc.models.lightglue_runtime_service import LightGlueServiceClient


from dafc.paths import PROJECT
ORCHESTRATOR_VERSION = "formal-miniwob-transitionguard-v2-external-artifacts"
FORMAL_MANIFEST_VERSION = "transitionguard-aaai27-manifest-v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def ensure_safe_component(value: str) -> str:
    if (
        not value
        or len(value) > 180
        or any(not (char.isalnum() or char in "-_.") for char in value)
    ):
        raise ValueError(f"unsafe experiment identifier: {value!r}")
    return value


def attempt_record_id(logical_id: str, attempt: int) -> str:
    ensure_safe_component(logical_id)
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    return f"{logical_id}--a{attempt:02d}"


def artifact_relative(phase: str, record_id: str) -> str:
    ensure_safe_component(record_id)
    return f"{phase}/attempts/{record_id}.json"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


class MiniWoBFormalExperiment:
    def __init__(
        self,
        manifest_path: Path,
        *,
        grounder_overrides: dict[
            str, Callable[[str, Path, str, str], dict[str, Any]]
        ]
        | None = None,
        match_provider_override: Callable[[Any, Any], dict[str, Any]]
        | None = None,
        strict_repository: bool = True,
    ):
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_json(self.manifest_path)
        self.manifest_sha256 = formal.sha256(self.manifest_path)
        self.grounder_overrides = grounder_overrides or {}
        self.match_provider_override = match_provider_override
        self.strict_repository = strict_repository
        self._model_health: dict[str, dict[str, Any]] = {}
        self._lightglue: LightGlueServiceClient | None = None
        self._lightglue_health: dict[str, Any] | None = None
        self.validate_manifest()
        self.root = Path(self.manifest["experiment_root"]).resolve()
        self.risk_certificate = live.load_risk_certificate(
            self.resolve_project_path(
                self.manifest["risk_certificate"]["path"]
            )
        )

    @property
    def is_formal(self) -> bool:
        return bool(self.manifest["paper_evidence"])

    def resolve_project_path(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()

    def validate_manifest(self) -> None:
        row = self.manifest
        if row.get("manifest_version") != FORMAL_MANIFEST_VERSION:
            raise RuntimeError("formal manifest version mismatch")
        if row.get("protocol_version") != formal.PROTOCOL_VERSION:
            raise RuntimeError("formal protocol version mismatch")
        if row.get("orchestrator_version") != ORCHESTRATOR_VERSION:
            raise RuntimeError("formal orchestrator version mismatch")
        if row.get("method_version") != live.GUARD.METHOD_VERSION:
            raise RuntimeError("manifest method version mismatch")
        if row.get("miniwob_runner_version") != live.RUNNER_VERSION:
            raise RuntimeError("manifest MiniWoB runner version mismatch")
        if not isinstance(row.get("paper_evidence"), bool):
            raise RuntimeError("manifest evidence scope is absent")

        tasks = tuple(row["tasks"])
        if not tasks or any(task not in live.TASKS for task in tasks):
            raise RuntimeError("manifest task cohort is invalid")
        if self.is_formal and tasks != live.TASKS:
            raise RuntimeError("formal manifest must retain the frozen eight tasks")
        source = row["source"]
        if source["model"] != "qwen25vl" or source["contract"] != "source":
            raise RuntimeError("source demonstrator boundary changed")
        if int(source["successes_per_task"]) <= 0:
            raise RuntimeError("source success target must be positive")
        cap = int(source["candidate_cap_per_task"])
        if cap <= 0:
            raise RuntimeError("source candidate cap must be positive")
        if self.is_formal:
            if int(source["successes_per_task"]) != formal.SOURCE_SUCCESSES_PER_TASK:
                raise RuntimeError("formal source success target changed")
            if cap != formal.SOURCE_CANDIDATE_CAP:
                raise RuntimeError("formal source candidate cap changed")
            expected_derivation = (
                'SHA256("transitionguard-aaai27-formal-v1|source|TASK|INDEX") '
                "first 64 bits masked to a positive signed 31-bit integer"
            )
            if source.get("seed_derivation") != expected_derivation:
                raise RuntimeError("formal source seed derivation changed")

            exclusions = row.get("development_seed_exclusions")
            if not isinstance(exclusions, dict):
                raise RuntimeError("formal development seed exclusions are absent")
            observed, observed_bindings = formal.development_seed_inventory(
                PROJECT / "validation", tasks
            )
            declared_by_task = {
                task: [int(value) for value in exclusions["by_task"][task]]
                for task in tasks
            }
            expected_by_task = {
                task: sorted(observed[task]) for task in tasks
            }
            if declared_by_task != expected_by_task:
                raise RuntimeError("formal development seed exclusions changed")
            if exclusions.get("source_bindings") != observed_bindings:
                raise RuntimeError("formal development seed inventory changed")
            inventory_payload = {
                "by_task": expected_by_task,
                "source_bindings": observed_bindings,
            }
            if exclusions.get("inventory_sha256") != formal.canonical_json_sha256(
                inventory_payload
            ):
                raise RuntimeError("formal development seed inventory hash mismatch")

        for task in tasks:
            candidates = source["candidates"][task]
            if len(candidates) != cap:
                raise RuntimeError(f"candidate cap mismatch for {task}")
            seeds = [int(candidate["seed"]) for candidate in candidates]
            if len(seeds) != len(set(seeds)):
                raise RuntimeError(f"duplicate source candidate seed for {task}")
            if self.is_formal:
                normalized = [
                    {
                        "derivation_index": int(candidate["derivation_index"]),
                        "seed": int(candidate["seed"]),
                    }
                    for candidate in candidates
                ]
                expected = formal.eligible_candidate_seeds(
                    task,
                    row["development_seed_exclusions"]["by_task"][task],
                    cap,
                )
                if normalized != expected:
                    raise RuntimeError(f"formal candidate derivation mismatch: {task}")

        target = row["target"]
        if tuple(target["policies"]) != (
            "always",
            "raw",
            "naive",
            "sg",
            "sgp",
            "full",
        ):
            raise RuntimeError("target policy cohort changed")
        if self.is_formal:
            if tuple(target["models"]) != ("qwen25vl", "uitars"):
                raise RuntimeError("formal target model order changed")
            if tuple(target["contracts"]) != (
                "native_zoom_125",
                "native_zoom_80",
                "native_affine_skew",
                "native_jpeg_stream",
                "native_narrow_reflow_124",
            ):
                raise RuntimeError("formal target contract cohort changed")
            if tuple(target["structural_contracts"]) != (
                "native_narrow_reflow_124",
            ):
                raise RuntimeError("formal structural contract boundary changed")
        for contract in target["contracts"]:
            live.contract_by_name(contract)
        if int(row["retrieval"]["minimum_distractors"]) < 7:
            raise RuntimeError("formal retrieval requires at least seven distractors")
        if self.is_formal and row["retrieval"].get(
            "development_distractor_records", []
        ):
            raise RuntimeError("formal retrieval cannot import development records")
        if int(row["attempt_policy"]["maximum_infrastructure_attempts"]) <= 0:
            raise RuntimeError("infrastructure attempt cap must be positive")
        if self.is_formal:
            if int(
                row["attempt_policy"]["maximum_infrastructure_attempts"]
            ) != 3:
                raise RuntimeError("formal infrastructure attempt cap changed")
            bootstrap = row["analysis"]["bootstrap"]
            if int(bootstrap["replicates"]) != 20_000:
                raise RuntimeError("formal bootstrap replicate count changed")
            if not isinstance(bootstrap.get("seed"), int) or isinstance(
                bootstrap.get("seed"), bool
            ):
                raise RuntimeError("formal bootstrap seed is invalid")
            required_namespaces = (
                "source",
                "target",
                "partial_progress",
                "dropped_action",
                "android_source",
                "android_target",
            )
            namespace_values = [
                ensure_safe_component(str(row["namespaces"][key]))
                for key in required_namespaces
            ]
            if len(namespace_values) != len(set(namespace_values)):
                raise RuntimeError("formal request namespaces are not unique")

            for kind in ("qwen25vl", "uitars"):
                service = row["model_services"][kind]
                health = service.get("preflight_health")
                checkpoint = service.get("checkpoint")
                attestation = service.get("checkpoint_attestation")
                if not isinstance(health, dict) or not isinstance(checkpoint, dict):
                    raise RuntimeError(f"formal model binding is absent: {kind}")
                if health.get("checkpoint") != checkpoint:
                    raise RuntimeError(f"formal health/checkpoint mismatch: {kind}")
                if not isinstance(attestation, dict):
                    raise RuntimeError(f"formal checkpoint attestation is absent: {kind}")
                expected_files = {
                    item["name"]: int(item["bytes"])
                    for item in checkpoint["weight_files"]
                }
                attested_files = {
                    item["name"]: int(item["bytes"])
                    for item in attestation["weight_files"]
                }
                if expected_files != attested_files or any(
                    len(str(item.get("sha256", ""))) != 64
                    for item in attestation["weight_files"]
                ):
                    raise RuntimeError(f"formal checkpoint attestation changed: {kind}")

            lightglue = row["lightglue_service"]
            if lightglue.get("preflight_health", {}).get("bindings") != lightglue.get(
                "bindings"
            ):
                raise RuntimeError("formal LightGlue binding is absent")
            formal.verify_external_file_bindings(row["external_files"])
            formal.verify_external_tree_bindings(row["external_trees"])
            formal.verify_external_git_bindings(row["external_git_repositories"])

        certificate_path = self.resolve_project_path(row["risk_certificate"]["path"])
        if formal.sha256(certificate_path) != row["risk_certificate"]["sha256"]:
            raise RuntimeError("risk certificate hash mismatch")
        certificate = live.load_risk_certificate(certificate_path)
        if certificate.certificate_id != row["risk_certificate"]["certificate_id"]:
            raise RuntimeError("risk certificate identity mismatch")

        for relative, expected in row.get("bound_files", {}).items():
            path = self.resolve_project_path(relative)
            if formal.sha256(path) != expected:
                raise RuntimeError(f"bound source hash mismatch: {relative}")

        root = Path(row["experiment_root"])
        if not root.is_absolute():
            raise RuntimeError("experiment root must be absolute")
        if self.is_formal:
            try:
                root.resolve().relative_to(PROJECT.resolve())
            except ValueError:
                pass
            else:
                raise RuntimeError("formal results must live outside the git tree")
            if not row.get("frozen_before_calls"):
                raise RuntimeError("formal manifest is not marked frozen")

    def verify_repository(self) -> None:
        if not self.strict_repository or not self.is_formal:
            return
        if git("status", "--porcelain"):
            raise RuntimeError("formal execution refuses a dirty repository")
        code_commit = str(self.manifest["code_commit"])
        git("merge-base", "--is-ancestor", code_commit, "HEAD")
        for relative, expected in self.manifest["bound_files"].items():
            if formal.sha256(self.resolve_project_path(relative)) != expected:
                raise RuntimeError(f"formal bound file changed: {relative}")

    def ensure_activation(self) -> dict[str, Any]:
        self.verify_repository()
        activation_path = self.root / "activation.json"
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)
        elif not activation_path.exists() and any(self.root.iterdir()):
            raise RuntimeError("nonempty experiment root lacks activation receipt")
        activation = {
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "paper_evidence": self.is_formal,
            "code_commit": self.manifest.get("code_commit"),
            "activation_git_head": git("rev-parse", "HEAD"),
            "activated_utc": utc_now(),
        }
        if activation_path.exists():
            existing = load_json(activation_path)
            stable_fields = (
                "orchestrator_version",
                "manifest_path",
                "manifest_sha256",
                "paper_evidence",
                "code_commit",
            )
            if any(existing.get(key) != activation.get(key) for key in stable_fields):
                raise RuntimeError("activation receipt does not match manifest")
            return existing
        formal.atomic_write_json(activation_path, activation)
        return activation

    def model_health(self, kind: str) -> dict[str, Any]:
        if kind in self._model_health:
            return self._model_health[kind]
        if kind in self.grounder_overrides:
            health = {
                "ready": True,
                "kind": kind,
                "development_grounder_override": True,
            }
        else:
            url = self.manifest["model_services"][kind]["url"]
            health = live.CLIENT.health(url)
            if not health.get("ready") or health.get("kind") != kind:
                raise RuntimeError(f"model service health mismatch: {kind}")
            expected = self.manifest["model_services"][kind].get(
                "service_version"
            )
            if expected and health.get("service_version") != expected:
                raise RuntimeError(f"model service version mismatch: {kind}")
            expected_runner = self.manifest["model_services"][kind].get(
                "base_runner_version"
            )
            if expected_runner and health.get("base_runner_version") != expected_runner:
                raise RuntimeError(f"model runner version mismatch: {kind}")
            expected_checkpoint = self.manifest["model_services"][kind].get(
                "checkpoint"
            )
            if expected_checkpoint and health.get("checkpoint") != expected_checkpoint:
                raise RuntimeError(f"model checkpoint binding mismatch: {kind}")
        self._model_health[kind] = health
        return health

    def grounder(self, kind: str):
        if kind in self.grounder_overrides:
            return self.grounder_overrides[kind]
        url = self.manifest["model_services"][kind]["url"]

        def call(
            request_id: str, image_path: Path, goal: str, history: str
        ) -> dict[str, Any]:
            return live.service_grounder(
                url, kind, request_id, image_path, goal, history
            )

        return call

    def match_provider(self):
        if self.match_provider_override is not None:
            return self.match_provider_override, {
                "ready": True,
                "service_version": "development-match-provider-override",
            }
        if self._lightglue is None:
            self._lightglue = LightGlueServiceClient(
                self.manifest["lightglue_service"]["url"]
            )
            self._lightglue_health = self._lightglue.health()
            if not self._lightglue_health.get("ready"):
                raise RuntimeError("LightGlue service is not ready")
            expected = self.manifest["lightglue_service"].get(
                "service_version"
            )
            if expected and self._lightglue_health.get("service_version") != expected:
                raise RuntimeError("LightGlue service version mismatch")
            expected_bindings = self.manifest["lightglue_service"].get("bindings")
            if expected_bindings and self._lightglue_health.get(
                "bindings"
            ) != expected_bindings:
                raise RuntimeError("LightGlue runtime binding mismatch")
        return self._lightglue.correspondences, dict(self._lightglue_health or {})

    def ledger_session(self, phase: str) -> formal.LedgerSession:
        ledger = formal.AppendOnlyLedger(
            self.root / phase / "ledger.jsonl", self.root
        )
        return formal.LedgerSession(ledger)

    @staticmethod
    def completed_rows(
        rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["status"] != "completed":
                continue
            logical_id = str(row["logical_id"])
            if logical_id in result:
                raise RuntimeError(f"multiple completed attempts: {logical_id}")
            result[logical_id] = row
        return result

    @staticmethod
    def attempts_for(
        rows: list[dict[str, Any]], logical_id: str
    ) -> list[dict[str, Any]]:
        attempts = [row for row in rows if row["logical_id"] == logical_id]
        attempts.sort(key=lambda row: int(row["attempt"]))
        expected = list(range(len(attempts)))
        observed = [int(row["attempt"]) for row in attempts]
        if observed != expected:
            raise RuntimeError(f"noncontiguous attempts for {logical_id}: {observed}")
        return attempts

    def append_attempt_artifact(
        self,
        session: formal.LedgerSession,
        *,
        phase: str,
        logical_id: str,
        attempt: int,
        status: str,
        artifact: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        relative = artifact.relative_to(self.root).as_posix()
        row = {
            "record_id": attempt_record_id(logical_id, attempt),
            "logical_id": logical_id,
            "phase": phase,
            "attempt": attempt,
            "status": status,
            "artifact": relative,
            "artifact_sha256": formal.sha256(artifact),
            "completed_utc": utc_now(),
            **metadata,
        }
        session.append(row)
        return row

    def orphan_attempt_directory(
        self,
        *,
        images_root: Path,
        run_id: str,
        task: str,
        seed: int,
        contract: str,
        kind: str,
        policy: str,
        attempt: int,
    ) -> Path:
        episode_id = (
            f"{run_id}--{task}--{seed}--{contract}--{kind}--{policy}--"
            f"a{attempt:02d}"
        )
        return images_root / episode_id

    def run_attempt(
        self,
        session: formal.LedgerSession,
        *,
        phase: str,
        logical_id: str,
        attempt: int,
        metadata: dict[str, Any],
        execute: Callable[[], dict[str, Any]],
        orphan_directory: Path,
    ) -> dict[str, Any]:
        record_id = attempt_record_id(logical_id, attempt)
        artifact = self.root / artifact_relative(phase, record_id)
        if artifact.exists():
            payload = load_json(artifact)
            formal_meta = payload.get("formal_attempt", {})
            if (
                formal_meta.get("record_id") != record_id
                or formal_meta.get("logical_id") != logical_id
                or int(formal_meta.get("attempt", -1)) != attempt
            ):
                raise RuntimeError("orphan artifact metadata mismatch")
            return self.append_attempt_artifact(
                session,
                phase=phase,
                logical_id=logical_id,
                attempt=attempt,
                status=str(formal_meta["status"]),
                artifact=artifact,
                metadata=metadata,
            )
        if orphan_directory.exists():
            payload = {
                "title": "Interrupted formal attempt with orphan image directory",
                "paper_evidence": self.is_formal,
                "formal_attempt": {
                    "record_id": record_id,
                    "logical_id": logical_id,
                    "attempt": attempt,
                    "status": "infrastructure_failure",
                },
                "orphan_image_directory": str(orphan_directory),
                "reason": "attempt_directory_exists_without_atomic_artifact",
            }
            formal.atomic_write_json(artifact, payload)
            return self.append_attempt_artifact(
                session,
                phase=phase,
                logical_id=logical_id,
                attempt=attempt,
                status="infrastructure_failure",
                artifact=artifact,
                metadata=metadata,
            )
        try:
            payload = execute()
            status = "completed"
        except Exception as error:  # retained as an immutable infrastructure attempt
            payload = {
                "title": "Formal infrastructure failure",
                "paper_evidence": self.is_formal,
                "error_type": type(error).__name__,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            status = "infrastructure_failure"
        payload["paper_evidence"] = self.is_formal
        payload["formal_attempt"] = {
            "record_id": record_id,
            "logical_id": logical_id,
            "attempt": attempt,
            "status": status,
            "manifest_sha256": self.manifest_sha256,
        }
        formal.atomic_write_json(artifact, payload)
        return self.append_attempt_artifact(
            session,
            phase=phase,
            logical_id=logical_id,
            attempt=attempt,
            status=status,
            artifact=artifact,
            metadata=metadata,
        )

    def source_logical_id(self, task: str, candidate: dict[str, Any]) -> str:
        return ensure_safe_component(
            f"source--{task}--c{int(candidate['derivation_index']):04d}--"
            f"s{int(candidate['seed'])}"
        )

    def source_success_rows(
        self, session: formal.LedgerSession
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.completed_rows(session.rows).values():
            if row.get("phase") != "source":
                continue
            episode = load_json(self.root / row["artifact"])
            if episode.get("success") is True:
                result[str(row["task"])].append(row)
        for rows in result.values():
            rows.sort(key=lambda item: int(item["candidate_derivation_index"]))
        return result

    def next_source_job(
        self, session: formal.LedgerSession
    ) -> tuple[str, dict[str, Any], int] | None:
        completed = self.completed_rows(session.rows)
        successes = self.source_success_rows(session)
        required = int(self.manifest["source"]["successes_per_task"])
        maximum_attempts = int(
            self.manifest["attempt_policy"]["maximum_infrastructure_attempts"]
        )
        tasks = tuple(self.manifest["tasks"])
        cap = int(self.manifest["source"]["candidate_cap_per_task"])
        for candidate_position in range(cap):
            for task in tasks:
                if len(successes.get(task, [])) >= required:
                    continue
                candidate = self.manifest["source"]["candidates"][task][
                    candidate_position
                ]
                logical_id = self.source_logical_id(task, candidate)
                if logical_id in completed:
                    continue
                attempts = self.attempts_for(session.rows, logical_id)
                if len(attempts) >= maximum_attempts:
                    raise RuntimeError(
                        f"source infrastructure attempt cap reached: {logical_id}"
                    )
                return task, candidate, len(attempts)
        missing = {
            task: required - len(successes.get(task, []))
            for task in tasks
            if len(successes.get(task, [])) < required
        }
        if missing:
            raise RuntimeError(f"source candidate cap exhausted: {missing}")
        return None

    def execute_source(
        self, task: str, candidate: dict[str, Any], attempt: int
    ) -> dict[str, Any]:
        kind = str(self.manifest["source"]["model"])
        health = self.model_health(kind)
        episode = live.run_episode(
            task=task,
            seed=int(candidate["seed"]),
            contract=live.contract_by_name("source"),
            policy="always",
            kind=kind,
            output_dir=self.root / "source" / "images",
            grounder=self.grounder(kind),
            run_id=self.manifest["namespaces"]["source"],
            attempt=attempt,
        )
        episode["services"] = {"model": health}
        episode["formal_source_candidate"] = dict(candidate)
        return episode

    def run_source(self, limit: int | None = None) -> dict[str, Any]:
        self.ensure_activation()
        session = self.ledger_session("source")
        attempts_run = 0
        while limit is None or attempts_run < limit:
            job = self.next_source_job(session)
            if job is None:
                break
            task, candidate, attempt = job
            logical_id = self.source_logical_id(task, candidate)
            images_root = self.root / "source" / "images"
            orphan = self.orphan_attempt_directory(
                images_root=images_root,
                run_id=self.manifest["namespaces"]["source"],
                task=task,
                seed=int(candidate["seed"]),
                contract="source",
                kind=self.manifest["source"]["model"],
                policy="always",
                attempt=attempt,
            )
            self.run_attempt(
                session,
                phase="source",
                logical_id=logical_id,
                attempt=attempt,
                metadata={
                    "task": task,
                    "seed": int(candidate["seed"]),
                    "candidate_derivation_index": int(
                        candidate["derivation_index"]
                    ),
                    "model": self.manifest["source"]["model"],
                },
                execute=lambda task=task, candidate=candidate, attempt=attempt: self.execute_source(
                    task, candidate, attempt
                ),
                orphan_directory=orphan,
            )
            attempts_run += 1
        selected = self.freeze_selected_sources(session)
        return {
            "attempts_run": attempts_run,
            "ledger_rows": len(session.rows),
            "selected_sources_ready": selected is not None,
            "status": self.source_status(session),
        }

    def source_status(
        self, session: formal.LedgerSession | None = None
    ) -> dict[str, Any]:
        session = session or self.ledger_session("source")
        successes = self.source_success_rows(session)
        completed = self.completed_rows(session.rows)
        return {
            "required_successes_per_task": int(
                self.manifest["source"]["successes_per_task"]
            ),
            "successes": {
                task: len(successes.get(task, [])) for task in self.manifest["tasks"]
            },
            "completed_candidates": sum(
                row.get("phase") == "source" for row in completed.values()
            ),
            "infrastructure_failures": sum(
                row["status"] == "infrastructure_failure" for row in session.rows
            ),
            "ledger_rows": len(session.rows),
        }

    def freeze_selected_sources(
        self, session: formal.LedgerSession
    ) -> dict[str, Any] | None:
        required = int(self.manifest["source"]["successes_per_task"])
        successes = self.source_success_rows(session)
        if any(
            len(successes.get(task, [])) < required
            for task in self.manifest["tasks"]
        ):
            return None
        records = []
        for task in self.manifest["tasks"]:
            for rank, row in enumerate(successes[task][:required]):
                artifact = self.root / row["artifact"]
                episode = load_json(artifact)
                records.append(
                    {
                        "source_cluster_id": f"{task}--r{rank:02d}--s{row['seed']}",
                        "task": task,
                        "rank_within_task": rank,
                        "seed": int(row["seed"]),
                        "candidate_derivation_index": int(
                            row["candidate_derivation_index"]
                        ),
                        "goal": episode["goal"],
                        "episode_id": episode["episode_id"],
                        "artifact": row["artifact"],
                        "artifact_sha256": row["artifact_sha256"],
                        "model_calls": episode["model_calls"],
                    }
                )
        payload = {
            "title": "Frozen successful source bank",
            "paper_evidence": self.is_formal,
            "manifest_sha256": self.manifest_sha256,
            "selection_rule": "first successful completed candidates by derivation index",
            "records": records,
            "records_sha256": formal.canonical_json_sha256(records),
            "frozen_utc": utc_now(),
        }
        path = self.root / "source" / "selected_sources.json"
        if path.exists():
            existing = load_json(path)
            if existing.get("records_sha256") != payload["records_sha256"]:
                raise RuntimeError("frozen selected source bank changed")
            return existing
        formal.atomic_write_json(path, payload)
        return payload

    def selected_sources(self) -> dict[str, Any]:
        path = self.root / "source" / "selected_sources.json"
        if not path.exists():
            raise RuntimeError("source bank is not complete and frozen")
        selected = load_json(path)
        if selected["manifest_sha256"] != self.manifest_sha256:
            raise RuntimeError("selected source manifest mismatch")
        if formal.canonical_json_sha256(selected["records"]) != selected[
            "records_sha256"
        ]:
            raise RuntimeError("selected source bank hash mismatch")
        return selected

    def retrieval_for_source(
        self, source: dict[str, Any], records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        minimum = int(self.manifest["retrieval"]["minimum_distractors"])
        candidates = [
            row
            for row in records
            if row["source_cluster_id"] != source["source_cluster_id"]
            and row["goal"] != source["goal"]
        ]
        extras = self.manifest["retrieval"].get(
            "development_distractor_records", []
        )
        candidates.extend(extras)
        candidates.sort(
            key=lambda row: formal.canonical_json_sha256(
                {
                    "protocol": formal.PROTOCOL_VERSION,
                    "query": source["source_cluster_id"],
                    "candidate": row["source_cluster_id"],
                }
            )
        )
        distractors = candidates[:minimum]
        if len(distractors) < minimum:
            raise RuntimeError("not enough different-goal source distractors")
        experience_records = [
            formal.ExperienceRecord(
                record_id=source["source_cluster_id"],
                goal=source["goal"],
                task=source["task"],
                seed=int(source["seed"]),
                source_episode=source["artifact"],
            )
        ] + [
            formal.ExperienceRecord(
                record_id=row["source_cluster_id"],
                goal=row["goal"],
                task=row["task"],
                seed=int(row["seed"]),
                source_episode=row["artifact"],
            )
            for row in distractors
        ]
        matched, receipt = formal.ExperienceIndex(
            experience_records
        ).retrieve_exact(source["goal"], minimum_distractors=minimum)
        if [row.record_id for row in matched] != [source["source_cluster_id"]]:
            raise RuntimeError("query-local experience bank is not unique")
        receipt["bank_scope"] = (
            "paired source trajectory plus deterministic different-goal distractors"
        )
        receipt["same_goal_source_records_omitted"] = sum(
            row["source_cluster_id"] != source["source_cluster_id"]
            and row["goal"] == source["goal"]
            for row in records
        )
        return receipt

    def prepare_target_plan(self) -> dict[str, Any]:
        self.ensure_activation()
        selected = self.selected_sources()
        records = list(selected["records"])
        jobs = []
        for source in records:
            retrieval = self.retrieval_for_source(source, records)
            for contract in self.manifest["target"]["contracts"]:
                for model in self.manifest["target"]["models"]:
                    for policy in self.manifest["target"]["policies"]:
                        logical_id = ensure_safe_component(
                            f"target--{source['source_cluster_id']}--{contract}--"
                            f"{model}--{policy}"
                        )
                        job = {
                            "logical_id": logical_id,
                            "source_cluster_id": source["source_cluster_id"],
                            "source_artifact": source["artifact"],
                            "source_artifact_sha256": source["artifact_sha256"],
                            "task": source["task"],
                            "seed": int(source["seed"]),
                            "goal": source["goal"],
                            "contract": contract,
                            "structural_stress": contract
                            in self.manifest["target"]["structural_contracts"],
                            "model": model,
                            "policy": policy,
                            "experience_retrieval": retrieval,
                        }
                        job["order_key"] = formal.canonical_json_sha256(
                            {
                                "protocol": formal.PROTOCOL_VERSION,
                                "target_job": logical_id,
                            }
                        )
                        jobs.append(job)
        jobs.sort(key=lambda row: (row["model"], row["order_key"]))
        payload = {
            "title": "Frozen target execution plan",
            "paper_evidence": self.is_formal,
            "manifest_sha256": self.manifest_sha256,
            "selected_sources_sha256": selected["records_sha256"],
            "ordering": "SHA256(protocol_version | logical target job)",
            "jobs": jobs,
            "jobs_sha256": formal.canonical_json_sha256(jobs),
            "frozen_utc": utc_now(),
        }
        path = self.root / "target" / "target_plan.json"
        if path.exists():
            existing = load_json(path)
            if existing.get("jobs_sha256") != payload["jobs_sha256"]:
                raise RuntimeError("target plan changed after freeze")
            return existing
        formal.atomic_write_json(path, payload)
        return payload

    def target_plan(self) -> dict[str, Any]:
        path = self.root / "target" / "target_plan.json"
        if not path.exists():
            raise RuntimeError("target plan is not prepared")
        plan = load_json(path)
        if plan["manifest_sha256"] != self.manifest_sha256:
            raise RuntimeError("target plan manifest mismatch")
        if formal.canonical_json_sha256(plan["jobs"]) != plan["jobs_sha256"]:
            raise RuntimeError("target plan hash mismatch")
        return plan

    def execute_target(self, job: dict[str, Any], attempt: int) -> dict[str, Any]:
        kind = str(job["model"])
        policy = str(job["policy"])
        health = self.model_health(kind)
        memory_goal = None
        memory = None
        if policy != "always":
            source_path = self.root / job["source_artifact"]
            if formal.sha256(source_path) != job["source_artifact_sha256"]:
                raise RuntimeError("target source artifact hash mismatch")
            memory_goal, memory = live.load_memory(source_path)
            if memory_goal != job["goal"]:
                raise RuntimeError("target plan/source goal mismatch")
        match_provider = None
        lightglue_health = None
        if policy in live.GUARDED_POLICIES:
            match_provider, lightglue_health = self.match_provider()
        episode = live.run_episode(
            task=job["task"],
            seed=int(job["seed"]),
            contract=live.contract_by_name(job["contract"]),
            policy=policy,
            kind=kind,
            output_dir=self.root / "target" / kind / "images",
            grounder=self.grounder(kind),
            run_id=self.manifest["namespaces"]["target"],
            attempt=attempt,
            memory_goal=memory_goal,
            memory=memory,
            match_provider=match_provider,
            geometry_frontend=(
                lightglue_health.get("service_version")
                if lightglue_health is not None
                else (
                    "not_applicable_sequential_replay"
                    if policy in live.SEQUENTIAL_POLICIES
                    else "not_applicable_always_ground"
                )
            ),
            risk_certificate=self.risk_certificate if policy == "full" else None,
            require_risk_certificate=policy == "full",
            score_postcondition_for_ablations=policy in live.GUARDED_POLICIES,
        )
        episode["services"] = {
            "model": health,
            "lightglue": lightglue_health,
        }
        episode["formal_target_job"] = {
            key: value for key, value in job.items() if key != "order_key"
        }
        return episode

    def run_target(self, model: str, limit: int | None = None) -> dict[str, Any]:
        self.ensure_activation()
        if model not in self.manifest["target"]["models"]:
            raise ValueError("model is outside the target plan")
        plan = self.target_plan()
        jobs = [job for job in plan["jobs"] if job["model"] == model]
        phase = f"target/{model}"
        session = self.ledger_session(phase)
        maximum_attempts = int(
            self.manifest["attempt_policy"]["maximum_infrastructure_attempts"]
        )
        attempts_run = 0
        for job in jobs:
            if limit is not None and attempts_run >= limit:
                break
            completed = self.completed_rows(session.rows)
            logical_id = job["logical_id"]
            if logical_id in completed:
                continue
            attempts = self.attempts_for(session.rows, logical_id)
            if len(attempts) >= maximum_attempts:
                raise RuntimeError(
                    f"target infrastructure attempt cap reached: {logical_id}"
                )
            attempt = len(attempts)
            images_root = self.root / "target" / model / "images"
            orphan = self.orphan_attempt_directory(
                images_root=images_root,
                run_id=self.manifest["namespaces"]["target"],
                task=job["task"],
                seed=int(job["seed"]),
                contract=job["contract"],
                kind=model,
                policy=job["policy"],
                attempt=attempt,
            )
            self.run_attempt(
                session,
                phase=phase,
                logical_id=logical_id,
                attempt=attempt,
                metadata={
                    "model": model,
                    "task": job["task"],
                    "seed": int(job["seed"]),
                    "source_cluster_id": job["source_cluster_id"],
                    "contract": job["contract"],
                    "policy": job["policy"],
                    "structural_stress": bool(job["structural_stress"]),
                },
                execute=lambda job=job, attempt=attempt: self.execute_target(
                    job, attempt
                ),
                orphan_directory=orphan,
            )
            attempts_run += 1
        return {
            "model": model,
            "attempts_run": attempts_run,
            "status": self.target_status(model, plan, session),
        }

    def target_status(
        self,
        model: str,
        plan: dict[str, Any] | None = None,
        session: formal.LedgerSession | None = None,
    ) -> dict[str, Any]:
        plan = plan or self.target_plan()
        phase = f"target/{model}"
        session = session or self.ledger_session(phase)
        jobs = [job for job in plan["jobs"] if job["model"] == model]
        completed = self.completed_rows(session.rows)
        return {
            "planned_jobs": len(jobs),
            "completed_jobs": sum(job["logical_id"] in completed for job in jobs),
            "infrastructure_failures": sum(
                row["status"] == "infrastructure_failure" for row in session.rows
            ),
            "ledger_rows": len(session.rows),
        }

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "manifest_sha256": self.manifest_sha256,
            "paper_evidence": self.is_formal,
            "source": self.source_status(),
        }
        target_plan = self.root / "target" / "target_plan.json"
        if target_plan.exists():
            plan = self.target_plan()
            result["target"] = {
                model: self.target_status(model, plan)
                for model in self.manifest["target"]["models"]
            }
        else:
            result["target"] = "not_prepared"
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("run-source")
    source.add_argument("--limit", type=int)
    subparsers.add_parser("prepare-target")
    target = subparsers.add_parser("run-target")
    target.add_argument("--model", choices=("qwen25vl", "uitars"), required=True)
    target.add_argument("--limit", type=int)
    subparsers.add_parser("status")
    args = parser.parse_args()

    experiment = MiniWoBFormalExperiment(args.manifest)
    if args.command == "run-source":
        result = experiment.run_source(args.limit)
    elif args.command == "prepare-target":
        plan = experiment.prepare_target_plan()
        result = {
            "jobs": len(plan["jobs"]),
            "jobs_sha256": plan["jobs_sha256"],
        }
    elif args.command == "run-target":
        result = experiment.run_target(args.model, args.limit)
    elif args.command == "status":
        result = experiment.status()
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
