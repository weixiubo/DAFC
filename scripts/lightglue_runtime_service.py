"""Compatibility entry point; maintained implementation: dafc.models.lightglue_runtime_service."""
import os
from pathlib import Path
from importlib import import_module
os.environ.setdefault("DAFC_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
_implementation = import_module('dafc.models.lightglue_runtime_service')
def __getattr__(name):
    return getattr(_implementation, name)
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
