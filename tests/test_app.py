import asyncio
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from cube_solver.python_app.server import app as app_module


class CameraLoopLifecycleTests(unittest.TestCase):
    def test_lifespan_starts_and_stops_camera_loop(self) -> None:
        with patch.object(app_module.camera_loop, "start") as start_mock, patch.object(app_module.camera_loop, "stop") as stop_mock:
            async def run_lifespan() -> None:
                async with app_module.lifespan(app_module.app):
                    pass

            asyncio.run(run_lifespan())

        start_mock.assert_called_once()
        stop_mock.assert_called_once()


class CameraStreamTests(unittest.TestCase):
    def test_camera_stream_endpoint_returns_jpeg(self) -> None:
        with patch.object(app_module.camera_loop, "start"), patch.object(app_module.camera_loop, "stop"), patch.object(app_module.camera_loop, "get_latest_frame_bytes", return_value=b"jpeg-bytes"):
            client = TestClient(app_module.app)
            response = client.get("/camera/stream")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.content, b"jpeg-bytes")

    def test_camera_loop_broadcasts_scan_event_payload(self) -> None:
        captured_payloads = []

        async def fake_broadcast(payload):
            captured_payloads.append(payload)

        def fake_read() -> tuple[bool, np.ndarray]:
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        with patch.object(app_module.camera_loop, "running", True), patch.object(app_module.camera_loop, "cap") as mock_cap, patch.object(app_module.scanner, "process_frame", return_value={"type": "face_scanned", "face": "U", "colors": ["white"] * 9}), patch.object(app_module, "broadcaster") as broadcaster_mock:
            mock_cap.read.side_effect = fake_read
            broadcaster_mock.broadcast.side_effect = fake_broadcast
            app_module.camera_loop.run()

        self.assertEqual(len(captured_payloads), 1)
        self.assertEqual(captured_payloads[0]["type"], "scan_event")
        self.assertEqual(captured_payloads[0]["face"], "U")


if __name__ == "__main__":
    unittest.main()
