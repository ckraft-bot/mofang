from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from cube_solver.python_app.cube.state import CubeState, CubeStateError
    from cube_solver.python_app.capture.color_detection import ColorDetector
else:
    from ..cube.state import CubeState, CubeStateError
    from .color_detection import ColorDetector


class FaceScanStateMachine:
    """Tracks which face is being scanned and accumulates scanned stickers."""

    def __init__(self, cube_state: CubeState | None = None, detector: ColorDetector | None = None):
        self.cube_state = cube_state or CubeState()
        self.detector = detector or ColorDetector()
        self.face_order = ["U", "R", "F", "D", "L", "B"]
        self.current_index = 0
        self.current_face = self.face_order[0]
        self.scan_complete = False
        self.last_event: dict[str, Any] | None = None

    def reset(self) -> None:
        self.cube_state.reset()
        self.current_index = 0
        self.current_face = self.face_order[0]
        self.scan_complete = False
        self.last_event = None

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        if self.scan_complete:
            return {"type": "scan_complete", "face": None, "colors": []}

        colors = self.detector.sample_face_grid(frame)
        if len(colors) != 9:
            colors = ["unknown"] * 9

        scanned_face = self.current_face

        try:
            self.cube_state.set_face(scanned_face, colors)
        except CubeStateError as exc:
            return {"type": "scan_error", "face": scanned_face, "error": str(exc)}

        self.current_index += 1
        if self.current_index >= len(self.face_order):
            self.scan_complete = True
            event = {
                "type": "scan_complete",
                "face": scanned_face,
                "colors": colors,
                "remaining_faces": [],
                "completed_faces": self.face_order,
                "message": "scan complete",
                "scan_state": {face: None for face in self.face_order},
            }
            event["scan_state"][scanned_face] = colors
        else:
            self.current_face = self.face_order[self.current_index]
            surface_color = colors[4] if len(colors) == 9 else "unknown"
            event = {
                "type": "face_scanned",
                "face": scanned_face,
                "colors": colors,
                "remaining_faces": self.face_order[self.current_index + 1 :],
                "completed_faces": self.face_order[: self.current_index],
                "message": f"{surface_color} surface scanned, go to the next surface",
                "scan_state": {face: None for face in self.face_order},
                "next_face": self.current_face,
            }
            event["scan_state"][scanned_face] = colors

        self.last_event = event
        return event

    def render_overlay(self, frame: np.ndarray, message: str = "") -> np.ndarray:
        display = frame.copy()
        cv2.putText(display, f"Scan face: {self.current_face}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, message or "Align the face and press scan", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return display
