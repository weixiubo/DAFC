"""Explicit receipt paths and SDK discovery, resolved only when requested."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def receipt_path(value: str, *, base: Path) -> Path:
    return (base / Path(value).expanduser()).resolve()


def adb_executable(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.environ.get("DAFC_ADB")
    if configured:
        return Path(configured).expanduser().resolve()
    sdk = Path(os.environ.get("ANDROID_HOME", str(Path.home() / "Android/Sdk")))
    return Path(shutil.which("adb") or sdk / "platform-tools/adb")
