"""Write a deterministic toy image pair into an explicitly selected directory."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    directory = parser.parse_args().directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(585001).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    for name in ("source.png", "target.png"):
        path = directory / name
        if path.exists():
            raise FileExistsError(path)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not encode {path}")
    print(directory)


if __name__ == "__main__":
    main()
