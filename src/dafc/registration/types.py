"""Registration service boundary."""

from collections.abc import Callable
from typing import Any

import numpy as np

PointMatcher = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, dict[str, Any]]]
