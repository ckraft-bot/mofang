# Mofang - Rubik's Cube Solver via Computer Vision

Scan a physical Rubik's cube with a webcam, solve it, and watch the
step-by-step solution as a synced 3D animation + move list in a browser
dashboard.

## Project structure

- `cube_solver/python_app/` -- OpenCV webcam scanner + color detection +
  Kociemba solver. Runs a local FastAPI/WebSocket backend.
- `cube_solver/web_dashboard/` -- Vite + Three.js frontend for the 3D cube
  preview and move list UI.

## Current status

Implemented so far:
- `cube_solver/python_app/cube/state.py` -- builds and validates a 54-char
  Kociemba facelet string.
- `cube_solver/python_app/cube/solver.py` -- wraps the `kociemba` solver
  and returns move objects for animation.
- `cube_solver/python_app/server/app.py` -- FastAPI backend with health,
  WebSocket, and solve endpoints.
- `cube_solver/web_dashboard/` -- simple dashboard shell that connects to
  the backend.

## Run the app

### 1) Install Python dependencies

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

### 2) Start the backend

From the repository root, run:

```powershell
python run_backend.py
```

This starts the FastAPI backend on:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/

### 3) Start the frontend

In a second terminal:

```powershell
cd cube_solver/web_dashboard
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Then open the frontend at:

- http://127.0.0.1:5173/

If port 5173 is busy, Vite will choose the next available port and report it in the terminal.

### 4) Open the dashboard

Open the frontend URL in your browser. The dashboard will connect to the backend over WebSocket and show the live camera preview, scan messages, and the 3D cube preview.

## Quick solver sanity check

```powershell
cd cube_solver/python_app
python -c "
from cube_solver.python_app.cube.state import CubeState
from cube_solver.python_app.cube.solver import solve_cube_state

known = 'BBURUDBFUFFFRRFUUFLULUFUDLRRDBBDBDBLUDDFLLRRBRLLLBRDDF'
faces = {'U': known[0:9], 'R': known[9:18], 'F': known[18:27],
         'D': known[27:36], 'L': known[36:45], 'B': known[45:54]}
cs = CubeState()
for face, s in faces.items():
    cs.set_face(face, list(s))
moves = solve_cube_state(cs)
print(' '.join(m.notation for m in moves))
"
```
