"""Compatibility entry point; maintained implementation: dafc.adapters.browser.miniwob_agent_harness."""
import os
from pathlib import Path
from importlib import import_module
os.environ.setdefault("DAFC_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
_implementation = import_module('dafc.adapters.browser.miniwob_agent_harness')
def __getattr__(name):
    return getattr(_implementation, name)
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
