"""Compatibility entry point; maintained implementation: dafc.evaluation.analyze_android_recovered_source_followup_v2_integral_coordinate_repaired."""
import os
from pathlib import Path
from importlib import import_module
os.environ.setdefault("DAFC_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
_implementation = import_module('dafc.evaluation.analyze_android_recovered_source_followup_v2_integral_coordinate_repaired')
def __getattr__(name):
    return getattr(_implementation, name)
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
