from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dafc.adapters.android import elements

ADB_SERIAL = "emulator-5554"
DEFAULT_DENSITY = 420


@dataclass(frozen=True)
class ContractSpec:
    contract_id: str
    density: int = DEFAULT_DENSITY
    font_scale: float = 1.0
    jpeg_quality: int | None = None


CONTRACTS = (
    ContractSpec("identity"),
    ContractSpec("density-480", density=480),
    ContractSpec("font-1.15", font_scale=1.15),
    ContractSpec("jpeg-70", jpeg_quality=70),
)


def pixels(state: Any) -> np.ndarray:
    return cv2.cvtColor(state.pixels, cv2.COLOR_RGB2BGR)


def observe(image: np.ndarray, contract: ContractSpec) -> np.ndarray:
    if contract.jpeg_quality is None:
        return image.copy()
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, contract.jpeg_quality])
    if not ok:
        raise RuntimeError("could not JPEG-transform target observation")
    transformed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if transformed is None or transformed.shape != image.shape:
        raise RuntimeError("JPEG target observation changed image shape")
    return transformed


def element_label(element: Any) -> str | None:
    return element.content_description or element.text


def has_label(state: Any, target: str) -> bool:
    return any(
        (
            target in {element.text, element.content_description}
            and element.bbox_pixels is not None
            and (element.is_visible is not False)
            and (element.is_enabled is not False)
            for element in state.ui_elements
        )
    )


def wait_state(env: Any, *, required_label: str | None = None, timeout_s: float = 8.0) -> Any:
    deadline = time.monotonic() + timeout_s
    best = None
    while time.monotonic() < deadline:
        state = env.get_state(wait_to_stabilize=True)
        if best is None or len(state.ui_elements) > len(best.ui_elements):
            best = state
        if len(state.ui_elements) > 5 and (required_label is None or has_label(state, required_label)):
            return state
        time.sleep(0.5)
    if best is None:
        raise RuntimeError("AndroidWorld returned no state")
    return best


def click_label(env: Any, state: Any, target: str) -> dict[str, Any]:
    if not has_label(state, target):
        state = wait_state(env, required_label=target)
    return elements.click_element(env, state, target)


def open_app(env: Any, app_name: str, first_label: str) -> Any:
    from android_world.env import json_action

    env.execute_action(json_action.JSONAction(action_type="open_app", app_name=app_name))
    time.sleep(2.0)
    return wait_state(env, required_label=first_label)


@dataclass(frozen=True)
class AndroidDevice:
    executable: Path
    serial: str = "emulator-5554"

    def adb(self, *args: str) -> str:
        return subprocess.run(
            [str(self.executable), "-s", self.serial, *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def restore_display(self) -> None:
        self.adb("shell", "wm", "density", "reset")
        self.adb("shell", "settings", "delete", "system", "font_scale")
        self.adb("shell", "cmd", "uimode", "night", "no")
        time.sleep(1.0)

    def apply_contract(self, contract: ContractSpec) -> None:
        self.restore_display()
        if contract.density != DEFAULT_DENSITY:
            self.adb("shell", "wm", "density", str(contract.density))
        if contract.font_scale != 1.0:
            self.adb("shell", "settings", "put", "system", "font_scale", f"{contract.font_scale:.2f}")
        time.sleep(1.0)

    def display_receipt(self) -> dict[str, str]:
        return {
            "density": self.adb("shell", "wm", "density"),
            "font_scale": self.adb("shell", "settings", "get", "system", "font_scale"),
            "night_mode": self.adb("shell", "cmd", "uimode", "night"),
        }
