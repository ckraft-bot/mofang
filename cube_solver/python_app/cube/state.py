"""
Cube state representation.

We use the standard Kociemba facelet string format: a 54-character string,
9 characters per face, in the order U R F D L B. Each character is one of
U R F D L B representing the color of that sticker by which face's center
has that color (i.e. color is identified by the center-sticker letter of
the face it belongs to on a solved cube).

Facelet index layout per face (as required by the `kociemba` package):

        0  1  2
        3  4  5      <- face is read left-to-right, top-to-bottom
        6  7  8

Face order in the 54-char string: U(0-8) R(9-17) F(18-26) D(27-35) L(36-44) B(45-53)

Index 4, 13, 22, 31, 40, 49 are the six center stickers -- these never move
and define which letter (U/R/F/D/L/B) corresponds to which physical color.
"""

from collections import Counter
from itertools import product

FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
FACELET_COUNT = 54
STICKERS_PER_FACE = 9
CENTER_INDICES = {face: FACE_ORDER.index(face) * 9 + 4 for face in FACE_ORDER}
FACE_ROTATION_PERMUTATIONS = {
    0: (0, 1, 2, 3, 4, 5, 6, 7, 8),
    1: (6, 3, 0, 7, 4, 1, 8, 5, 2),
    2: (8, 7, 6, 5, 4, 3, 2, 1, 0),
    3: (2, 5, 8, 1, 4, 7, 0, 3, 6),
}


class CubeStateError(ValueError):
    """Raised when a scanned cube state is invalid or incomplete."""


class CubeState:
    """
    Accumulates scanned faces (each a list of 9 physical color labels, e.g.
    "red", "white", ...) and produces a validated Kociemba facelet string.
    """

    def __init__(self):
        # maps face letter -> list of 9 raw color labels (strings), or None if unscanned
        self._faces: dict[str, list[str] | None] = {f: None for f in FACE_ORDER}

    def set_face(self, face: str, colors: list[str]) -> None:
        if face not in FACE_ORDER:
            raise CubeStateError(f"Unknown face '{face}'. Must be one of {FACE_ORDER}.")
        if len(colors) != STICKERS_PER_FACE:
            raise CubeStateError(f"Face '{face}' must have exactly 9 colors, got {len(colors)}.")
        self._faces[face] = list(colors)

    def is_fully_scanned(self) -> bool:
        return all(v is not None for v in self._faces.values())

    def missing_faces(self) -> list[str]:
        return [f for f, v in self._faces.items() if v is None]

    def reset(self) -> None:
        self._faces = {f: None for f in FACE_ORDER}

    @staticmethod
    def _rotate_face(colors: list[str], turns: int) -> list[str]:
        return [colors[index] for index in FACE_ROTATION_PERMUTATIONS[turns % 4]]

    def _build_facelet_string(self, faces: dict[str, list[str]], color_to_letter: dict[str, str]) -> str:
        facelets = []
        for face in FACE_ORDER:
            for raw_color in faces[face]:
                if raw_color not in color_to_letter:
                    raise CubeStateError(
                        f"Sticker color '{raw_color}' on face {face} doesn't match any "
                        f"known center color. Recalibrate or rescan."
                    )
                facelets.append(color_to_letter[raw_color])
        return "".join(facelets)

    def _build_color_to_letter_map(self) -> dict[str, str]:
        color_to_letter: dict[str, str] = {}
        for face in FACE_ORDER:
            center_color = self._faces[face][4]
            if center_color in color_to_letter:
                raise CubeStateError(
                    f"Two faces have the same center color '{center_color}' "
                    f"({color_to_letter[center_color]} and {face}). Check scan/calibration."
                )
            color_to_letter[center_color] = face

        if len(color_to_letter) != 6:
            raise CubeStateError("Expected exactly 6 distinct center colors, got a different count.")

        return color_to_letter

    def to_facelet_string(self, auto_orient: bool = False) -> str:
        """
        Convert accumulated raw color scans into the 54-char Kociemba facelet
        string. Raises CubeStateError with a human-readable reason if the
        scan is incomplete or physically invalid.
        """
        if not self.is_fully_scanned():
            raise CubeStateError(
                f"Cannot solve: missing faces {self.missing_faces()}. Scan all 6 faces first."
            )

        color_to_letter = self._build_color_to_letter_map()

        if not auto_orient:
            facelet_string = self._build_facelet_string({face: list(colors) for face, colors in self._faces.items() if colors is not None}, color_to_letter)
            self._validate_facelet_string(facelet_string)
            return facelet_string

        try:
            import kociemba
        except ImportError as exc:  # pragma: no cover - defensive fallback
            raise CubeStateError("kociemba solver is required for orientation normalization.") from exc

        best_candidate: str | None = None
        best_score: int | None = None

        for rotation_choice in product(range(4), repeat=len(FACE_ORDER)):
            rotated_faces = {
                face: self._rotate_face(self._faces[face], turns)
                for face, turns in zip(FACE_ORDER, rotation_choice)
            }
            candidate = self._build_facelet_string(rotated_faces, color_to_letter)

            try:
                self._validate_facelet_string(candidate)
                kociemba.solve(candidate)
            except ValueError:
                continue

            score = sum(min(turns, 4 - turns) for turns in rotation_choice)
            if best_score is None or score < best_score:
                best_candidate = candidate
                best_score = score
                if score == 0:
                    break

        if best_candidate is None:
            raise CubeStateError(
                "Could not normalize face orientations. Make sure each face is scanned with the sticker grid upright."
            )

        return best_candidate

    @staticmethod
    def _validate_facelet_string(facelet_string: str) -> None:
        if len(facelet_string) != FACELET_COUNT:
            raise CubeStateError(
                f"Facelet string must be {FACELET_COUNT} characters, got {len(facelet_string)}."
            )

        counts = Counter(facelet_string)
        bad = {letter: n for letter, n in counts.items() if n != 9}
        if bad:
            raise CubeStateError(
                f"Each color must appear exactly 9 times. Off counts: {bad}. "
                f"This usually means a misread sticker during scanning."
            )

        # Centers must match their own face position (U-center must be 'U', etc.)
        for face in FACE_ORDER:
            idx = CENTER_INDICES[face]
            if facelet_string[idx] != face:
                raise CubeStateError(
                    f"Center sticker mismatch on face {face}: expected '{face}', "
                    f"got '{facelet_string[idx]}'. Check face scan order."
                )
