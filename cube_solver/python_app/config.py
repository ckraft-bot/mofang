import os
from pathlib import Path


class AppConfig:
    host = os.getenv("MOFANG_HOST", "0.0.0.0")
    port = int(os.getenv("MOFANG_PORT", "8000"))
    camera_index = int(os.getenv("MOFANG_CAMERA_INDEX", "0"))
    calibration_path = Path(os.getenv("MOFANG_CALIBRATION_PATH", Path(__file__).with_name("calibration.json")))
    face_order = ["U", "R", "F", "D", "L", "B"]
    default_colors = ["white", "red", "green", "yellow", "orange", "blue"]
