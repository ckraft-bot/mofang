"""
Wraps the `kociemba` two-phase-algorithm solver and turns its raw move
string into a structured, step-by-step move list that's easy for the
JS dashboard to animate and display one move at a time.
"""

from dataclasses import dataclass

import kociemba

from .state import CubeStateError


@dataclass(frozen=True)
class Move:
    face: str          # one of U R F D L B
    turns: int          # 1 = clockwise 90°, 2 = 180°, 3 = counter-clockwise 90° (i.e. "'")
    notation: str        # original notation, e.g. "R", "U2", "F'"

    @property
    def is_clockwise(self) -> bool:
        return self.turns == 1

    @property
    def is_double(self) -> bool:
        return self.turns == 2

    @property
    def is_counter_clockwise(self) -> bool:
        return self.turns == 3

    @property
    def degrees(self) -> int:
        """Rotation angle for animation purposes, always expressed as the
        shortest signed rotation: +90, -90, or 180."""
        return {1: 90, 2: 180, 3: -90}[self.turns]


class SolveError(RuntimeError):
    """Raised when the underlying solver fails (e.g. unsolvable/invalid cube)."""


def _parse_move(token: str) -> Move:
    face = token[0]
    if len(token) == 1:
        turns = 1
    elif token[1] == "2":
        turns = 2
    elif token[1] == "'":
        turns = 3
    else:
        raise SolveError(f"Unrecognized move token '{token}'")
    return Move(face=face, turns=turns, notation=token)


def solve_facelet_string(facelet_string: str, max_depth: int = 24) -> list[Move]:
    """
    Solve a 54-char Kociemba facelet string and return an ordered list of Move.
    Raises SolveError if the cube is invalid/unsolvable.
    """
    try:
        raw_solution = kociemba.solve(facelet_string, max_depth=max_depth)
    except ValueError as e:
        # kociemba raises ValueError for malformed/invalid states
        raise SolveError(f"Cube state rejected by solver: {e}") from e

    if not raw_solution or raw_solution.strip() == "":
        raise SolveError("Solver returned an empty solution -- cube may already be solved.")

    tokens = raw_solution.strip().split()
    return [_parse_move(t) for t in tokens]


def solve_cube_state(cube_state) -> list[Move]:
    """Convenience wrapper: takes a CubeState (see state.py), normalizes orientation, validates, and solves it."""
    facelet_string = cube_state.to_facelet_string(auto_orient=True)  # raises CubeStateError if incomplete/invalid
    return solve_facelet_string(facelet_string)


def moves_to_json(moves: list[Move]) -> list[dict]:
    """Serialize a Move list into JSON-friendly dicts for the WebSocket bridge."""
    return [
        {
            "index": i,
            "notation": m.notation,
            "face": m.face,
            "turns": m.turns,
            "degrees": m.degrees,
        }
        for i, m in enumerate(moves)
    ]
