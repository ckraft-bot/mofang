from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from cube_solver.python_app.calibration import CalibrationManager
    from cube_solver.python_app.capture.color_detection import ColorDetector
    from cube_solver.python_app.capture.scanner import FaceScanStateMachine
    from cube_solver.python_app.cube.solver import solve_cube_state, moves_to_json
    from cube_solver.python_app.cube.state import CubeState
    from cube_solver.python_app.config import AppConfig
    from cube_solver.python_app.server.broadcast import EventBroadcaster
else:
    from ..calibration import CalibrationManager
    from ..capture.color_detection import ColorDetector
    from ..capture.scanner import FaceScanStateMachine
    from ..cube.solver import solve_cube_state, moves_to_json
    from ..cube.state import CubeState
    from ..config import AppConfig
    from .broadcast import EventBroadcaster

@asynccontextmanager
async def lifespan(app: FastAPI):
    camera_loop.start()
    try:
        yield
    finally:
        camera_loop.stop()


app = FastAPI(title="Mofang", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
broadcaster = EventBroadcaster()
config = AppConfig()
calibration = CalibrationManager(config.calibration_path)
detector = ColorDetector(config.calibration_path)
scanner = FaceScanStateMachine(CubeState(), detector)


@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse("<h1>Mofang backend ready</h1><p>Open the dashboard frontend to connect.</p>")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "camera_open": camera_loop.is_open(), "scan_complete": scanner.scan_complete}


@app.post("/scan/reset")
async def reset_scan() -> dict[str, Any]:
    scanner.reset()
    await broadcaster.broadcast({"type": "scan_reset"})
    return {"status": "reset"}


@app.post("/scan/next")
async def scan_next() -> dict[str, Any]:
    return {"status": "ready", "face": scanner.current_face}


@app.post("/solve")
async def solve() -> dict[str, Any]:
    try:
        moves = solve_cube_state(scanner.cube_state)
        await broadcaster.broadcast({"type": "solution_ready", "moves": moves_to_json(moves)})
        return {"status": "solved", "moves": moves_to_json(moves)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await broadcaster.connect(websocket)
    try:
        await websocket.send_json({"type": "connected", "message": "dashboard connected"})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)


@app.post("/calibrate")
async def calibrate() -> dict[str, Any]:
    return {"status": "ok", "profile": calibration.as_dict()}


@app.get("/camera/stream")
async def camera_stream() -> Response:
    frame_bytes = camera_loop.get_latest_frame_bytes()
    if frame_bytes is None:
        return Response(status_code=204)
    return Response(content=frame_bytes, media_type="image/jpeg")


class CameraLoop:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(config.camera_index)
        self.running = False
        self.thread: threading.Thread | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_lock = threading.Lock()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_latest_frame_bytes(self) -> bytes | None:
        with self.latest_frame_lock:
            frame = self.latest_frame
        if frame is None:
            return None
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return None
        return buffer.tobytes()

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self.latest_frame_lock:
                self.latest_frame = frame
            event = scanner.process_frame(frame)
            if event.get("type") in {"face_scanned", "scan_complete"}:
                try:
                    asyncio.run(broadcaster.broadcast({"type": "scan_event", **event}))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(broadcaster.broadcast({"type": "scan_event", **event}))
                    finally:
                        loop.close()


camera_loop = CameraLoop()


def start_server() -> None:
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


def start_background_services() -> None:
    camera_loop.start()
    threading.Thread(target=start_server, daemon=True).start()
