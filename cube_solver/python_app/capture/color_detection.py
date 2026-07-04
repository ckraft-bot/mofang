from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ColorDetector:
    """Simple HSV-based color classifier with a JSON calibration profile."""

    _HSV_FALLBACKS = {
        "white": ((0, 0, 200), (0, 0, 255)),
        "yellow": ((20, 100, 100), (40, 255, 255)),
        "blue": ((90, 80, 80), (140, 255, 255)),
        "green": ((40, 80, 80), (90, 255, 255)),
        "red": ((0, 80, 80), (10, 255, 255)),
        "orange": ((10, 120, 120), (25, 255, 255)),
    }

    def __init__(self, calibration_path: str | Path | None = None):
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.profile: dict[str, list[float]] = self._load_profile()

    def _load_profile(self) -> dict[str, list[float]]:
        if self.calibration_path is None or not self.calibration_path.exists():
            return {}
        try:
            data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: [float(v) for v in values] for k, values in data.items() if isinstance(values, list)}
        except json.JSONDecodeError:
            return {}
        return {}

    def save_profile(self) -> None:
        if self.calibration_path is None:
            return
        self.calibration_path.write_text(json.dumps(self.profile, indent=2), encoding="utf-8")

    def classify(self, hsv: np.ndarray | tuple[int, int, int]) -> str:
        if isinstance(hsv, tuple):
            hsv_array = np.array(hsv, dtype=np.float32)
        else:
            hsv_array = np.asarray(hsv, dtype=np.float32)

        if self.profile:
            best_name = "unknown"
            best_distance = float("inf")
            for color_name, reference in self.profile.items():
                ref = np.asarray(reference, dtype=np.float32)
                distance = float(np.linalg.norm(hsv_array - ref))
                if distance < best_distance:
                    best_distance = distance
                    best_name = color_name
            return best_name

        for color_name, ranges in self._HSV_FALLBACKS.items():
            lower = np.array(ranges[0], dtype=np.float32)
            upper = np.array(ranges[1], dtype=np.float32)
            if np.all(hsv_array >= lower) and np.all(hsv_array <= upper):
                return color_name
        return "unknown"

    def sample_face_grid(self, frame: np.ndarray, grid_size: int = 3) -> list[str]:
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        colors: list[str] = []
        for row in range(grid_size):
            for col in range(grid_size):
                x = int((col + 0.5) * width / grid_size)
                y = int((row + 0.5) * height / grid_size)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y, x]
                colors.append(self.classify(tuple(int(v) for v in hsv)))
        return colors
