"""Compatibility entry point; maintained implementation: dafc.evaluation.audit_transitionguard_postcheck_baselines."""
import os
from pathlib import Path
from importlib import import_module
os.environ.setdefault("DAFC_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
_implementation = import_module('dafc.evaluation.audit_transitionguard_postcheck_baselines')
def __getattr__(name):
    return getattr(_implementation, name)
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
