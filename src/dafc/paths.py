"""Explicit experiment workspace; independent of the package install location."""
import os
from pathlib import Path
PROJECT = Path(os.environ.get("DAFC_PROJECT_ROOT", Path.cwd())).resolve()
