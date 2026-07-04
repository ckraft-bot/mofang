from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CalibrationManager:
    """A small helper for saving HSV values for each face color."""

    def __init__(self, calibration_path: str | Path | None = None):
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.profile: dict[str, list[float]] = self.load_profile()

    def load_profile(self) -> dict[str, list[float]]:
        if self.calibration_path is None or not self.calibration_path.exists():
            return {}
        return json.loads(self.calibration_path.read_text(encoding="utf-8"))

    def save_profile(self) -> None:
        if self.calibration_path is None:
            return
        self.calibration_path.write_text(json.dumps(self.profile, indent=2), encoding="utf-8")

    def calibrate_from_frame(self, frame: np.ndarray, color_name: str) -> dict[str, list[float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        center_y, center_x = height // 2, width // 2
        h, s, v = hsv[center_y, center_x]
        self.profile[color_name] = [float(h), float(s), float(v)]
        self.save_profile()
        return self.profile

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile}
