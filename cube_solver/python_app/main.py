from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cube_solver.python_app.server.app import start_background_services
else:
    from .server.app import start_background_services


if __name__ == "__main__":
    start_background_services()
