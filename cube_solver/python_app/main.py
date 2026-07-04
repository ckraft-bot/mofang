from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cube_solver.python_app.capture.scanner import FaceScanStateMachine
    from cube_solver.python_app.config import AppConfig
    from cube_solver.python_app.cube.solver import SolveError, solve_cube_state
    from cube_solver.python_app.cube.state import CubeStateError
else:
    from .capture.scanner import FaceScanStateMachine
    from .config import AppConfig
    from .cube.solver import SolveError, solve_cube_state
    from .cube.state import CubeStateError


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


def run_local_solver() -> int:
    config = AppConfig()
    scanner = FaceScanStateMachine()
    cap = cv2.VideoCapture(config.camera_index)
    window_name = "Mofang Local Solver"

    if not cap.isOpened():
        print(f"Unable to open camera index {config.camera_index}")
        return 1

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    message = "Align U face in grid and press C to capture"
    solution_notations: list[str] = []
    solution_index = 0
    calibration_colors = config.default_colors
    calibration_index = 1

    def blend_hsv(existing: list[float] | None, sample: list[float], alpha: float = 0.35) -> list[float]:
        if not existing:
            return sample

        h0, s0, v0 = existing
        h1, s1, v1 = sample
        delta_h = ((h1 - h0 + 90.0) % 180.0) - 90.0
        blended_h = (h0 + (alpha * delta_h)) % 180.0
        blended_s = ((1.0 - alpha) * s0) + (alpha * s1)
        blended_v = ((1.0 - alpha) * v0) + (alpha * v1)
        return [float(blended_h), float(blended_s), float(blended_v)]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                message = "Camera frame read failed"
                continue

            overlay = scanner.render_overlay(frame, message)

            if solution_notations:
                step_text = f"Step {solution_index + 1}/{len(solution_notations)}: {solution_notations[solution_index]}"
                _put_text_with_shadow(overlay, step_text, (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2)
                _put_text_with_shadow(
                    overlay,
                    "N: next step, P: previous step",
                    (12, 98),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )

            active_calibration = calibration_colors[calibration_index]
            _put_text_with_shadow(
                overlay,
                f"Calibrate: {active_calibration.upper()} (1-6 select, K sample, 0 clear)",
                (12, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            _put_text_with_shadow(
                overlay,
                "Rescan by center color: W/G/B/Y/O/E then press C",
                (12, 146),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                2,
            )

            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key).lower() if 0 <= key <= 255 else ""

            if key_char == "q":
                break

            if key_char == "r":
                scanner.reset()
                solution_notations = []
                solution_index = 0
                message = "Scan reset. Align U face and press C"
                continue

            if key_char in ("1", "2", "3", "4", "5", "6"):
                calibration_index = int(key_char) - 1
                message = f"Selected calibration color: {calibration_colors[calibration_index]}"
                continue

            if key_char == "k":
                points = scanner._grid_points(frame.shape)
                hsv = scanner.detector.sample_hsv_point(frame, points[4], patch_radius=12)
                color_name = calibration_colors[calibration_index]
                old = scanner.detector.profile.get(color_name)
                scanner.detector.profile[color_name] = blend_hsv(old, [float(hsv[0]), float(hsv[1]), float(hsv[2])])
                scanner.detector.save_profile()
                h, s, v = scanner.detector.profile[color_name]
                message = f"Saved {color_name} HSV: ({h:.1f}, {s:.1f}, {v:.1f})"
                continue

            if key_char == "0":
                scanner.detector.profile = {}
                scanner.detector.save_profile()
                message = "Calibration cleared. Using fallback detection"
                continue

            color_select_keys = {
                "w": "white",
                "g": "green",
                "b": "blue",
                "y": "yellow",
                "o": "orange",
                "e": "red",
            }
            if key_char in color_select_keys:
                color_name = color_select_keys[key_char]
                face = scanner.select_face_by_center_color(color_name)
                if face is None:
                    message = f"No scanned face has center color '{color_name}' yet"
                else:
                    message = f"Selected {color_name} center face ({face}). Press C to rescan"
                continue

            if key_char == "c":
                event = scanner.capture_face(frame)
                message = event.get("message", "face captured")
                completed_faces = len(event.get("completed_faces", []))
                if completed_faces:
                    message = f"{message} | Surfaces {completed_faces}/6"
                if event.get("type") == "scan_complete":
                    message = "All faces captured. Press S to solve"
                continue

            if key_char == "s":
                if not scanner.scan_complete:
                    message = f"Scan incomplete. Remaining: {', '.join(scanner.cube_state.missing_faces())}"
                    continue
                try:
                    moves = solve_cube_state(scanner.cube_state)
                    solution_notations = [m.notation for m in moves]
                    solution_index = 0
                    message = f"Solution ready: {len(solution_notations)} moves"
                    print("Solution:")
                    for idx, notation in enumerate(solution_notations, start=1):
                        print(f"{idx:02d}. {notation}")
                except (CubeStateError, SolveError) as exc:
                    message = f"Solve failed: {exc}"
                continue

            if key_char == "n" and solution_notations:
                solution_index = min(solution_index + 1, len(solution_notations) - 1)
                message = f"Next step: {solution_notations[solution_index]}"
                continue

            if key_char == "p" and solution_notations:
                solution_index = max(solution_index - 1, 0)
                message = f"Previous step: {solution_notations[solution_index]}"
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run_local_solver())
