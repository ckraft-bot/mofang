# Mofang - Rubik's Cube Solver via Computer Vision

Scan a physical Rubik's cube with a webcam, solve it, and watch the
step-by-step solution locally with OpenCV overlays and move notations.

## Project structure

- `cube_solver/python_app/` -- OpenCV webcam scanner + color detection +
  Kociemba solver.

## Current status

Implemented so far:
- `cube_solver/python_app/cube/state.py` -- builds and validates a 54-char
  Kociemba facelet string.
- `cube_solver/python_app/cube/solver.py` -- wraps the `kociemba` solver
  and returns move objects with notation.

## Solver algs

| Algorithm | Strategy | Speed | Best for |
| --- | --- | --- | --- |
| **Kociemba (Two‑Phase)** | Uses group theory to reduce cube to a restricted subgroup (G1), then solves | Very fast | Real‑time solvers and practical near‑optimal solutions |

## Run the local OpenCV solver

### 1) Install Python dependencies

From the repository root:

```powershell
python -m pip install -r cube_solver/python_app/requirements.txt
```

### 2) Start local scanner + solver

From the repository root, run:

```powershell
python -m cube_solver.python_app.main
```

Controls in the OpenCV window:

- `C` capture current face (order: U R F D L B)
- `S` solve after all faces are captured
- `N` next solution step
- `P` previous solution step
- `R` reset scan
- `1..6` choose calibration color (`white, yellow, blue, green, red, orange`)
- `K` sample center sticker and save calibration
- `0` clear calibration profile
- `Q` quit

The terminal prints the full notation sequence (for example `R U R' U'`) and
the window shows one move at a time.

Calibration tip:

- For red, sample under neutral lighting and press `K` 2-3 times to average noise.
- Keep the center sticker fully inside the guide box while calibrating.
- The app assumes the standard Rubik's Cube orientation: U white, R red, F green, D yellow, L orange, B blue.

