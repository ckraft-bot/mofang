from __future__ import annotations

import sys
from pathlib import Path

import cv2

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


def run_local_solver() -> int:
    config = AppConfig()
    scanner = FaceScanStateMachine()
    cap = cv2.VideoCapture(config.camera_index)

    if not cap.isOpened():
        print(f"Unable to open camera index {config.camera_index}")
        return 1

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
                cv2.putText(overlay, step_text, (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2)
                cv2.putText(overlay, "N: next step, P: previous step", (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            active_calibration = calibration_colors[calibration_index]
            cv2.putText(overlay, f"Calibrate: {active_calibration.upper()} (1-6 select, K sample, 0 clear)", (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("Mofang Local Solver", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                scanner.reset()
                solution_notations = []
                solution_index = 0
                message = "Scan reset. Align U face and press C"
                continue

            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                calibration_index = int(chr(key)) - 1
                message = f"Selected calibration color: {calibration_colors[calibration_index]}"
                continue

            if key == ord("k"):
                points = scanner._grid_points(frame.shape)
                hsv = scanner.detector.sample_hsv_point(frame, points[4], patch_radius=12)
                color_name = calibration_colors[calibration_index]
                old = scanner.detector.profile.get(color_name)
                scanner.detector.profile[color_name] = blend_hsv(old, [float(hsv[0]), float(hsv[1]), float(hsv[2])])
                scanner.detector.save_profile()
                h, s, v = scanner.detector.profile[color_name]
                message = f"Saved {color_name} HSV: ({h:.1f}, {s:.1f}, {v:.1f})"
                continue

            if key == ord("0"):
                scanner.detector.profile = {}
                scanner.detector.save_profile()
                message = "Calibration cleared. Using fallback detection"
                continue

            if key == ord("c"):
                event = scanner.capture_face(frame)
                message = event.get("message", "face captured")
                if event.get("type") == "scan_complete":
                    message = "All faces captured. Press S to solve"
                continue

            if key == ord("s"):
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

            if key == ord("n") and solution_notations:
                solution_index = min(solution_index + 1, len(solution_notations) - 1)
                message = f"Next step: {solution_notations[solution_index]}"
                continue

            if key == ord("p") and solution_notations:
                solution_index = max(solution_index - 1, 0)
                message = f"Previous step: {solution_notations[solution_index]}"
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run_local_solver())
