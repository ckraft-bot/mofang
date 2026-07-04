import unittest
from unittest.mock import MagicMock

import numpy as np

from cube_solver.python_app.capture.color_detection import ColorDetector
from cube_solver.python_app.capture.scanner import FaceScanStateMachine


class FaceScanStateMachineTests(unittest.TestCase):
    def test_scan_event_message_uses_center_color(self) -> None:
        scanner = FaceScanStateMachine()
        scanner.detector = MagicMock()
        scanner.detector.sample_face_grid.return_value = ["white"] * 9

        event = scanner.process_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertEqual(event["type"], "face_scanned")
        self.assertEqual(event["message"], "white surface scanned, go to the next surface")


class ColorDetectionTests(unittest.TestCase):
    def test_classify_without_profile_uses_hsv_fallback(self) -> None:
        detector = ColorDetector(calibration_path=None)

        self.assertEqual(detector.classify((120, 200, 220)), "blue")
        self.assertEqual(detector.classify((60, 220, 220)), "green")
        self.assertEqual(detector.classify((10, 220, 220)), "red")


if __name__ == "__main__":
    unittest.main()
