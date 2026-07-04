import unittest
from unittest.mock import MagicMock

import numpy as np

from cube_solver.python_app.capture.color_detection import ColorDetector
from cube_solver.python_app.capture.scanner import FaceScanStateMachine
from cube_solver.python_app.cube.state import CubeState


class FaceScanStateMachineTests(unittest.TestCase):
    def test_scan_event_message_uses_center_color(self) -> None:
        scanner = FaceScanStateMachine()
        scanner.detector = MagicMock()
        scanner.detector.sample_face_grid.return_value = ["white"] * 9

        event = scanner.capture_face(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertEqual(event["type"], "face_scanned")
        self.assertEqual(event["message"], "white surface scanned, go to the next surface")

    def test_process_frame_returns_preview_event(self) -> None:
        scanner = FaceScanStateMachine()
        scanner.detector = MagicMock()
        scanner.detector.sample_face_grid.return_value = ["green"] * 9

        event = scanner.process_frame(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(event["type"], "scan_preview")
        self.assertEqual(event["face"], "U")
        self.assertEqual(event["colors"], ["green"] * 9)


class ColorDetectionTests(unittest.TestCase):
    def test_classify_without_profile_uses_hsv_fallback(self) -> None:
        detector = ColorDetector(calibration_path=None)

        self.assertEqual(detector.classify((120, 200, 220)), "blue")
        self.assertEqual(detector.classify((60, 220, 220)), "green")
        self.assertEqual(detector.classify((10, 220, 220)), "red")
        self.assertEqual(detector.classify((175, 220, 220)), "red")
        self.assertEqual(detector.classify((12, 220, 220)), "orange")
        self.assertEqual(detector.classify((0, 40, 180)), "white")
        self.assertEqual(detector.classify((0, 95, 150)), "white")

    def test_classify_with_profile_handles_red_hue_wrap(self) -> None:
        detector = ColorDetector(calibration_path=None)
        detector.profile = {
            "red": [2.0, 210.0, 210.0],
            "orange": [18.0, 220.0, 220.0],
        }

        self.assertEqual(detector.classify((178, 215, 215)), "red")

    def test_incomplete_profile_does_not_override_fallback(self) -> None:
        detector = ColorDetector(calibration_path=None)
        detector.profile = {
            "blue": [120.0, 210.0, 210.0],
            "green": [60.0, 210.0, 210.0],
        }

        self.assertEqual(detector.classify((178, 220, 220)), "red")
        self.assertEqual(detector.classify((0, 50, 190)), "white")


class CubeStateOrientationTests(unittest.TestCase):
    def test_to_facelet_string_auto_orients_rotated_face_scan(self) -> None:
        known = "BBURUDBFUFFFRRFUUFLULUFUDLRRDBBDBDBLUDDFLLRRBRLLLBRDDF"
        faces = {
            "U": list(known[0:9]),
            "R": list(known[9:18]),
            "F": list(known[18:27]),
            "D": list(known[27:36]),
            "L": list(known[36:45]),
            "B": list(known[45:54]),
        }

        rotated_u = [faces["U"][6], faces["U"][3], faces["U"][0], faces["U"][7], faces["U"][4], faces["U"][1], faces["U"][8], faces["U"][5], faces["U"][2]]

        state = CubeState()
        state.set_face("U", rotated_u)
        state.set_face("R", faces["R"])
        state.set_face("F", faces["F"])
        state.set_face("D", faces["D"])
        state.set_face("L", faces["L"])
        state.set_face("B", faces["B"])

        self.assertEqual(state.to_facelet_string(auto_orient=True), known)


if __name__ == "__main__":
    unittest.main()
