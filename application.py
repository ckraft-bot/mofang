import cv2
import sys
import numpy as np
from datetime import datetime
import kociemba

CUBE_BOX = (200, 100, 400, 400)  # x, y, width, height
CONFIRM_FRAMES = 5  # consecutive frames to confirm face

# -----------------------------
# Detect face colors by sampling center of each sticker
# -----------------------------
import cv2
import numpy as np

def detect_face(frame):
    """
    Detect colors of a 3x3 cube face.
    Returns a list of 9 color names corresponding to stickers.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    face_colors = []

    # Define HSV ranges for common cube colors
    # Adjust ranges if your stickers differ
    colors_hsv = {
        "White": ((0, 0, 200), (180, 50, 255)),
        "Red": ((0, 100, 50), (10, 255, 255)),
        "Red2": ((170, 100, 50), (180, 255, 255)),  # wrap-around for red
        "Orange": ((10, 100, 50), (25, 255, 255)),
        "Yellow": ((25, 100, 50), (35, 255, 255)),
        "Green": ((35, 50, 50), (85, 255, 255)),
        "Blue": ((85, 50, 50), (130, 255, 255))
    }

    for row in range(3):
        for col in range(3):
            cx, cy = col*cell_w + cell_w//2, row*cell_h + cell_h//2
            pixel = hsv[cy, cx]
            matched = False

            for color, (lower, upper) in colors_hsv.items():
                lower = np.array(lower)
                upper = np.array(upper)
                if cv2.inRange(np.uint8([[pixel]]), lower, upper)[0][0] > 0:
                    # Merge Red2 with Red
                    face_colors.append("Red" if color.startswith("Red") else color)
                    matched = True
                    break

            if not matched:
                face_colors.append("Unknown")  # fallback if no match

    return face_colors


# -----------------------------
# Capture a single cube face
# -----------------------------
def capture_face(video, prompt_text="Show a face"):
    confirmed_faces = []

    while True:
        ret, frame = video.read()
        if not ret:
            print("Cannot read camera")
            sys.exit()

        # Crop to cube region
        x, y, w, h = CUBE_BOX
        cube_frame = frame[y:y+h, x:x+w]

        face = detect_face(cube_frame)

        if face:
            confirmed_faces.append(face)
            if len(confirmed_faces) >= CONFIRM_FRAMES:
                # Use np.unique for majority vote
                face_array = np.array(confirmed_faces)
                detected_face = []
                for i in range(9):
                    vals, counts = np.unique(face_array[:, i], return_counts=True)
                    detected_face.append(vals[np.argmax(counts)])

                cv2.putText(frame, "Captured!", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
                cv2.imshow("Cube Scanner", frame)
                cv2.waitKey(1000)  # pause 1 sec
                return detected_face
        else:
            confirmed_faces = []  # reset if face lost

        # Draw bounding box and prompt
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(frame, prompt_text, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        cv2.imshow("Cube Scanner", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            sys.exit()

# -----------------------------
# Main scanning loop
# -----------------------------
def main():
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    if not ret:
        print("Cannot read camera")
        sys.exit()

    notation_map = {"White":"U", "Yellow":"D", "Red":"F", "Orange":"B", "Blue":"L", "Green":"R"}
    scanned_faces = {}
    captured_centers = []

    print("Capture all 6 faces in any order. Center color identifies the face.")

    while len(scanned_faces) < 6:
        face = capture_face(video, prompt_text="Show any cube face")
        center_color = face[4]

        if center_color not in captured_centers:
            scanned_faces[center_color] = face
            captured_centers.append(center_color)
            print(f"Captured face with center {center_color}")
        else:
            print(f"Face {center_color} already captured. Show a different face.")

    video.release()
    cv2.destroyAllWindows()

    # Build Kociemba string in standard order: U, R, F, D, L, B
    face_order = ["U","R","F","D","L","B"]
    cube_str = ""
    for notation in face_order:
        center_color = [color for color, n in notation_map.items() if n == notation][0]
        face = scanned_faces[center_color]
        for color in face:
            cube_str += notation_map[color]

    print("Cube scanned:", cube_str)

    try:
        solution = kociemba.solve(cube_str)
        print("Solution:", solution)
    except Exception as e:
        print("Could not solve cube:", e)

if __name__ == "__main__":
    main()
