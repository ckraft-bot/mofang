from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from cube_solver.python_app.cube.state import CubeState, CubeStateError
    from cube_solver.python_app.capture.color_detection import ColorDetector
else:
    from ..cube.state import CubeState, CubeStateError
    from .color_detection import ColorDetector


_FONT_CACHE: dict[int, object] = {}


def _get_space_mono_font(pixel_size: int):
    if ImageFont is None:
        return None
    pixel_size = max(12, int(pixel_size))
    cached = _FONT_CACHE.get(pixel_size)
    if cached is not None:
        return cached

    candidates = [
        Path("C:/Windows/Fonts/SpaceMono-Bold.ttf"),
        Path("C:/Windows/Fonts/SpaceMono-Regular.ttf"),
        Path("C:/Users/Clair/AppData/Local/Microsoft/Windows/Fonts/SpaceMono-Bold.ttf"),
        Path("C:/Users/Clair/AppData/Local/Microsoft/Windows/Fonts/SpaceMono-Regular.ttf"),
        Path("/Library/Fonts/SpaceMono-Bold.ttf"),
        Path("/Library/Fonts/SpaceMono-Regular.ttf"),
        Path("/usr/share/fonts/truetype/spacemono/SpaceMono-Bold.ttf"),
        Path("/usr/share/fonts/truetype/spacemono/SpaceMono-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), pixel_size)
                _FONT_CACHE[pixel_size] = font
                return font
            except OSError:
                continue
    return None


def _put_text_with_shadow(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_face: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    shadow_offset: tuple[int, int] = (2, 2),
    line_type: int = cv2.LINE_AA,
) -> None:
    font = _get_space_mono_font(int((font_scale * 28) + (thickness * 2)))
    if font is not None and Image is not None and ImageDraw is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        text_top = int(org[1] - (font_scale * 26))
        shadow_top = text_top + shadow_offset[1]
        draw.text((org[0] + shadow_offset[0], shadow_top), text, font=font, fill=(shadow_color[2], shadow_color[1], shadow_color[0]))
        draw.text((org[0], text_top), text, font=font, fill=(color[2], color[1], color[0]))
        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return

    shadow_org = (org[0] + shadow_offset[0], org[1] + shadow_offset[1])
    cv2.putText(image, text, shadow_org, font_face, font_scale, shadow_color, thickness + 1, line_type)
    cv2.putText(image, text, org, font_face, font_scale, color, thickness, line_type)


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

    def _center_color_face_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for face, stickers in self.scan_state.items():
            if stickers and len(stickers) >= 5:
                mapping[stickers[4]] = face
        return mapping

    def select_face(self, face: str) -> bool:
        if face not in self.face_order:
            return False
        self.current_face = face
        self.current_index = self.face_order.index(face)
        return True

    def select_face_by_center_color(self, color_name: str) -> str | None:
        center_map = self._center_color_face_map()
        face = center_map.get(color_name)
        if face is None:
            return None
        self.select_face(face)
        return face

    def capture_face(self, frame: np.ndarray) -> dict[str, Any]:
        colors = self._sample_current_face(frame)
        if len(colors) != 9:
            colors = ["unknown"] * 9

        scanned_face = self.current_face
        was_previously_scanned = self.scan_state.get(scanned_face) is not None

        try:
            self.cube_state.set_face(scanned_face, colors)
        except CubeStateError as exc:
            return {"type": "scan_error", "face": scanned_face, "error": str(exc)}

        self.scan_state[scanned_face] = colors

        remaining_faces = [face for face in self.face_order if self.scan_state[face] is None]
        completed_faces = [face for face in self.face_order if self.scan_state[face] is not None]
        self.scan_complete = len(remaining_faces) == 0
        surface_color = colors[4] if len(colors) == 9 else "unknown"

        if was_previously_scanned:
            event = {
                "type": "face_rescanned",
                "face": scanned_face,
                "colors": colors,
                "remaining_faces": remaining_faces,
                "completed_faces": completed_faces,
                "message": f"Rescanned {scanned_face} (center: {surface_color})",
                "scan_state": self.scan_state,
                "next_face": self.current_face,
            }
        elif self.scan_complete:
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
            self.current_face = remaining_faces[0]
            self.current_index = self.face_order.index(self.current_face)
            event = {
                "type": "face_scanned",
                "face": scanned_face,
                "colors": colors,
                "remaining_faces": remaining_faces,
                "completed_faces": completed_faces,
                "message": f"{surface_color} surface scanned, go to the next surface",
                "scan_state": self.scan_state,
                "next_face": self.current_face,
            }

        self.last_event = event
        return event

    def _color_sticker_counts(self) -> tuple[dict[str, int], int]:
        tracked_colors = ["white", "green", "blue", "yellow", "orange", "red"]
        counts = {color_name: 0 for color_name in tracked_colors}
        scanned_stickers = 0
        for stickers in self.scan_state.values():
            if not stickers:
                continue
            scanned_stickers += len(stickers)
            for sticker_color in stickers:
                if sticker_color in counts:
                    counts[sticker_color] += 1
        return counts, scanned_stickers

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
        _put_text_with_shadow(
            display,
            f"Face: {self.current_face} ({face_step}/6)",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        _put_text_with_shadow(
            display,
            message or "Press C to capture face, R to reset, S to solve",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        counts, scanned_stickers = self._color_sticker_counts()
        scanned_faces = sum(1 for stickers in self.scan_state.values() if stickers is not None)
        panel_x = max(12, display.shape[1] - 210)
        panel_y = 24
        line_height = 24
        _put_text_with_shadow(
            display,
            f"Surfaces: {scanned_faces}/6",
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        _put_text_with_shadow(
            display,
            f"Stickers: {scanned_stickers}/54",
            (panel_x, panel_y + line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        for row, color_name in enumerate(["white", "green", "blue", "yellow", "orange", "red"]):
            label = f"{color_name} ({counts[color_name]}/9)"
            _put_text_with_shadow(
                display,
                label,
                (panel_x, panel_y + ((row + 2) * line_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        deficits: list[str] = []
        excesses: list[str] = []
        for color_name in ["white", "green", "blue", "yellow", "orange", "red"]:
            delta = counts[color_name] - 9
            if delta < 0:
                deficits.append(f"{color_name}+{-delta}")
            elif delta > 0:
                excesses.append(f"{color_name}-{delta}")

        status_y = panel_y + (8 * line_height)
        if scanned_faces < 6:
            _put_text_with_shadow(
                display,
                "Validate after 6/6 surfaces",
                (panel_x, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                2,
            )
        elif deficits or excesses:
            _put_text_with_shadow(
                display,
                "Re-scan needed",
                (panel_x, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (80, 200, 255),
                2,
            )
            if deficits:
                _put_text_with_shadow(
                    display,
                    f"Missing: {', '.join(deficits)}",
                    (panel_x, status_y + line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
            if excesses:
                _put_text_with_shadow(
                    display,
                    f"Extra: {', '.join(excesses)}",
                    (panel_x, status_y + (2 * line_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
        else:
            _put_text_with_shadow(
                display,
                "Color counts balanced",
                (panel_x, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 240, 120),
                2,
            )
        return display
