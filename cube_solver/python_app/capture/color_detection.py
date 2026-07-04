from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ColorDetector:
    """Simple HSV-based color classifier with a JSON calibration profile."""

    _HSV_FALLBACKS = {
        "white": [((0, 0, 180), (179, 70, 255))],
        "yellow": [((20, 90, 90), (40, 255, 255))],
        "blue": [((90, 80, 80), (140, 255, 255))],
        "green": [((40, 70, 70), (90, 255, 255))],
        "red": [((0, 80, 80), (10, 255, 255)), ((170, 80, 80), (179, 255, 255))],
        "orange": [((10, 120, 120), (25, 255, 255))],
    }
    _PROFILE_COLORS = {"white", "yellow", "blue", "green", "red", "orange"}

    def __init__(self, calibration_path: str | Path | None = None):
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.profile: dict[str, list[float]] = self._load_profile()

    @staticmethod
    def bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    @staticmethod
    def _hue_distance(h1: float, h2: float) -> float:
        diff = abs(h1 - h2)
        return min(diff, 180.0 - diff)

    def _profile_distance(self, hsv: np.ndarray, reference: np.ndarray) -> float:
        hue = self._hue_distance(float(hsv[0]), float(reference[0]))
        sat = abs(float(hsv[1]) - float(reference[1]))
        val = abs(float(hsv[2]) - float(reference[2]))
        return float((2.0 * hue) + (0.6 * sat) + (0.4 * val))

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

        h = float(hsv_array[0])
        s = float(hsv_array[1])
        v = float(hsv_array[2])

        # White can have higher saturation under warm lights; keep this tolerant.
        if v >= 100.0 and s <= 120.0:
            return "white"

        # Keep orange separate from red; orange commonly sits near hue 10-24.
        if s >= 80.0 and v >= 80.0 and 11.0 <= h <= 25.0:
            return "orange"

        # Red wraps around the HSV hue boundary near 0/179 in OpenCV.
        if s >= 70.0 and v >= 70.0 and (h <= 10.0 or h >= 170.0):
            return "red"

        # Only trust profile matching if all 6 cube colors are calibrated.
        if self.profile and self._PROFILE_COLORS.issubset(self.profile.keys()):
            best_name = "unknown"
            best_distance = float("inf")
            for color_name, reference in self.profile.items():
                ref = np.asarray(reference, dtype=np.float32)
                distance = self._profile_distance(hsv_array, ref)
                if distance < best_distance:
                    best_distance = distance
                    best_name = color_name
            if best_name != "unknown" and best_distance <= 120.0:
                return best_name

        for color_name, ranges in self._HSV_FALLBACKS.items():
            for lower_raw, upper_raw in ranges:
                lower = np.array(lower_raw, dtype=np.float32)
                upper = np.array(upper_raw, dtype=np.float32)
                if np.all(hsv_array >= lower) and np.all(hsv_array <= upper):
                    return color_name
        return "unknown"

    def sample_hsv_point(self, frame: np.ndarray, point: tuple[int, int], patch_radius: int = 10) -> np.ndarray:
        hsv_frame = self.bgr_to_hsv(frame)
        height, width = hsv_frame.shape[:2]
        x, y = point

        x1 = max(0, x - patch_radius)
        y1 = max(0, y - patch_radius)
        x2 = min(width, x + patch_radius + 1)
        y2 = min(height, y + patch_radius + 1)

        if x1 >= x2 or y1 >= y2:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        patch = hsv_frame[y1:y2, x1:x2]
        return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)

    def sample_face_grid(
        self,
        frame: np.ndarray,
        grid_size: int = 3,
        points: list[tuple[int, int]] | None = None,
        patch_radius: int = 10,
    ) -> list[str]:
        if frame is None or frame.size == 0:
            return []

        hsv_frame = self.bgr_to_hsv(frame)
        height, width = hsv_frame.shape[:2]

        if points is None:
            points = []
            for row in range(grid_size):
                for col in range(grid_size):
                    x = int((col + 0.5) * width / grid_size)
                    y = int((row + 0.5) * height / grid_size)
                    points.append((x, y))

        colors: list[str] = []
        for x, y in points:
            x1 = max(0, x - patch_radius)
            y1 = max(0, y - patch_radius)
            x2 = min(width, x + patch_radius + 1)
            y2 = min(height, y + patch_radius + 1)

            if x1 >= x2 or y1 >= y2:
                colors.append("unknown")
                continue

            patch = hsv_frame[y1:y2, x1:x2]
            mean_hsv = np.median(patch.reshape(-1, 3), axis=0)
            colors.append(self.classify(mean_hsv))
        return colors
