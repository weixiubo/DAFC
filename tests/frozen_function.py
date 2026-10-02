"""Load original function bodies only after verifying the complete frozen snapshot."""

import ast
import hashlib
import json
import tarfile
from pathlib import Path


def original_function(path, name, namespace):
    root = Path(__file__).resolve().parents[1] / "reproduce/frozen"
    metadata = json.loads((root / "SOURCE.json").read_text())
    archive = root / metadata["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
    with tarfile.open(archive) as bundle:
        source = bundle.extractfile(path).read().decode("utf-8-sig")
    fn = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(fn)
    exec(compile(ast.fix_missing_locations(module), "hash-verified-original", "exec"), namespace)
    return namespace[name]
