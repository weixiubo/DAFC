"""Load original function bodies only after verifying the complete frozen snapshot."""

import ast

from .reference_support import read_reference


def original_function(path, name, namespace):
    source = read_reference(path).decode("utf-8-sig")
    fn = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(fn)
    exec(compile(ast.fix_missing_locations(module), "hash-verified-original", "exec"), namespace)
    return namespace[name]
