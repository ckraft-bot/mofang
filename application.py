import cv2
import sys
import numpy as np
from scipy import stats
import time
import kociemba
from datetime import datetime

CUBE_BOX = (200, 100, 400, 400)  # x, y, width, height
CONFIRM_FRAMES = 5  # number of consecutive frames to confirm face

# -----------------------------
# Face Detection
# -----------------------------
def detect_face_colors(frame):
    # Define HSV ranges for your cube colors
    colors_hsv = {
        "white": ((0,0,180),(180,50,255)),
        "red": ((0,120,70),(10,255,255)),
        "orange": ((10,100,100),(25,255,255)),
        "yellow": ((25,100,100),(35,255,255)),
        "green": ((35,50,50),(85,255,255)),
        "blue": ((90,50,50),(130,255,255))
    }

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)
    centers = []

    for color_name, (lower, upper) in colors_hsv.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 3000:  # adjust threshold for your camera
                x, y, w, h = cv2.boundingRect(cnt)
                centers.append((x+w//2, y+h//2, color_name))
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    if len(centers) == 9:
        centers.sort(key=lambda c: (c[1], c[0]))  # sort by y, then x
        face_colors = [c[2] for c in centers]
        return face_colors
    return None

# -----------------------------
# Capture a single cube face with confirmation
# -----------------------------
def capture_face(video, videoWriter, prompt_text="Show a face"):
    confirmed_faces = []
    while True:
        ret, frame = video.read()
        if not ret:
            print("Cannot read video source")
            sys.exit()
        
        # Crop to cube region
        x, y, w, h = CUBE_BOX
        cube_frame = frame[y:y+h, x:x+w]

        face, _ = detect_face(cube_frame)

        if face is not None:
            confirmed_faces.append(face)
            if len(confirmed_faces) >= CONFIRM_FRAMES:
                # Majority vote
                face_array = np.array(confirmed_faces)
                detected_face = stats.mode(face_array, axis=0)[0][0]
                # Display confirmation
                cv2.putText(frame, "Captured! Show next face", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
                videoWriter.write(frame)
                cv2.imshow("Cube Scanner", frame)
                cv2.waitKey(1000)  # 1 second pause
                return detected_face
        else:
            confirmed_faces = []  # reset if face lost

        # Draw bounding box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(frame, prompt_text, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        videoWriter.write(frame)
        cv2.imshow("Cube Scanner", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            sys.exit()

# -----------------------------
# Main scanning loop
# -----------------------------
def main():
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    if not ret:
        print("Cannot read video source")
        sys.exit()

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter("media/cube_scan.avi", fourcc, 20.0, (w,h))

    scanned_faces = {}
    face_names = ["Up", "Down", "Front", "Back", "Left", "Right"]

    for name in face_names:
        scanned_faces[name] = capture_face(video, videoWriter, prompt_text=f"Show {name} center")

    video.release()
    videoWriter.release()
    cv2.destroyAllWindows()

    # Convert scanned_faces to Kociemba string
    # Assign unique color values to notations
    color_map = {}
    notations = ["U","D","F","B","L","R"]
    for i, face_name in enumerate(face_names):
        color_map[scanned_faces[face_name][4]] = notations[i]

    cube_str = ""
    for face_name in face_names:
        face = scanned_faces[face_name]
        for val in face:
            cube_str += color_map[val]

    print("Cube scanned:", cube_str)

    try:
        solution = kociemba.solve(cube_str)
        print("Solution:", solution)
    except Exception as e:
        print("Could not solve cube:", e)

if __name__ == "__main__":
    main()
