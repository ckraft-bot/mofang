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
    """Tracks cube face scanning progress and captures one face on demand."""

    def __init__(self, cube_state: CubeState | None = None, detector: ColorDetector | None = None):
        self.cube_state = cube_state or CubeState()
        self.detector = detector or ColorDetector()
        self.face_order = ["U", "R", "F", "D", "L", "B"]
        self.current_index = 0
        self.current_face = self.face_order[0]
        self.scan_complete = False
        self.last_event: dict[str, Any] | None = None
        self.scan_state: dict[str, list[str] | None] = {face: None for face in self.face_order}

    def reset(self) -> None:
        self.cube_state.reset()
        self.current_index = 0
        self.current_face = self.face_order[0]
        self.scan_complete = False
        self.last_event = None
        self.scan_state = {face: None for face in self.face_order}

    @staticmethod
    def _grid_points(frame_shape: tuple[int, int, int], grid_size: int = 3, size_ratio: float = 0.45) -> list[tuple[int, int]]:
        height, width = frame_shape[:2]
        side = int(min(width, height) * size_ratio)
        side = max(side, 90)

        origin_x = (width - side) // 2
        origin_y = (height - side) // 2
        cell = side / grid_size

        points: list[tuple[int, int]] = []
        for row in range(grid_size):
            for col in range(grid_size):
                x = int(origin_x + (col + 0.5) * cell)
                y = int(origin_y + (row + 0.5) * cell)
                points.append((x, y))
        return points

    def _sample_current_face(self, frame: np.ndarray) -> list[str]:
        points = self._grid_points(frame.shape)
        return self.detector.sample_face_grid(frame, points=points)

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        colors = self._sample_current_face(frame) if frame is not None and frame.size else []
        event = {
            "type": "scan_preview",
            "face": self.current_face,
            "colors": colors,
            "scan_complete": self.scan_complete,
        }
        self.last_event = event
        return event

    def capture_face(self, frame: np.ndarray) -> dict[str, Any]:
        if self.scan_complete:
            return {
                "type": "scan_complete",
                "face": None,
                "colors": [],
                "message": "all faces already captured",
                "remaining_faces": [],
                "completed_faces": self.face_order,
                "scan_state": self.scan_state,
            }

        colors = self._sample_current_face(frame)
        if len(colors) != 9:
            colors = ["unknown"] * 9

        scanned_face = self.current_face

        try:
            self.cube_state.set_face(scanned_face, colors)
        except CubeStateError as exc:
            return {"type": "scan_error", "face": scanned_face, "error": str(exc)}

        self.scan_state[scanned_face] = colors

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
                "scan_state": self.scan_state,
            }
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
                "scan_state": self.scan_state,
                "next_face": self.current_face,
            }

        self.last_event = event
        return event

    def render_overlay(self, frame: np.ndarray, message: str = "") -> np.ndarray:
        display = frame.copy()

        points = self._grid_points(display.shape)
        side = int(min(display.shape[1], display.shape[0]) * 0.45)
        origin_x = (display.shape[1] - side) // 2
        origin_y = (display.shape[0] - side) // 2
        cell = side // 3

        for row in range(4):
            y = origin_y + row * cell
            cv2.line(display, (origin_x, y), (origin_x + side, y), (255, 255, 255), 1)
        for col in range(4):
            x = origin_x + col * cell
            cv2.line(display, (x, origin_y), (x, origin_y + side), (255, 255, 255), 1)

        preview_colors = self.detector.sample_face_grid(display, points=points)
        palette = {
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "orange": (0, 165, 255),
            "unknown": (128, 128, 128),
        }
        for (x, y), color_name in zip(points, preview_colors):
            cv2.circle(display, (x, y), 8, palette.get(color_name, palette["unknown"]), -1)
            cv2.circle(display, (x, y), 10, (20, 20, 20), 1)

        face_step = min(self.current_index + 1, len(self.face_order))
        cv2.putText(display, f"Face: {self.current_face} ({face_step}/6)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, message or "Press C to capture face, R to reset, S to solve", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return display
