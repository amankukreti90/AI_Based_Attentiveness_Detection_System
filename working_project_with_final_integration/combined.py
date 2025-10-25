import cv2
import mediapipe as mp
import numpy as np
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model

# Load models
lip_model = load_model("lip_movement_model.h5")
eye_model = load_model("eye_state_model.h5")
head_model = load_model("head_movement.h5")


LABELS = ["still", "moving"]

# SQLite setup
conn = sqlite3.connect("attentiveness.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS full_attentiveness (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        lip_status TEXT,
        eye_status TEXT,
        head_status TEXT,
        lip_score REAL,
        eye_score REAL,
        head_score REAL,
        final_score REAL
    )
''')
conn.commit()
conn.close()

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(0)

def get_roi(image, points, w, h, margin=10):
    coords = [(int(pt.x * w), int(pt.y * h)) for pt in points]
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    x_min, x_max = max(min(xs)-margin, 0), min(max(xs)+margin, w)
    y_min, y_max = max(min(ys)-margin, 0), min(max(ys)+margin, h)
    return image[y_min:y_max, x_min:x_max]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    lip_score = eye_score = head_score = 0
    lip_status = eye_status = head_status = "Unknown"

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # ---- Lip Detection ----
        lip_points = [landmarks[i] for i in [61, 291, 78, 308, 13, 14, 17, 0]]
        mouth_roi = get_roi(frame, lip_points, w, h)

        if mouth_roi.size != 0:
            mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
            mouth_resized = cv2.resize(mouth_gray, (64, 64)) / 255.0
            mouth_input = mouth_resized.reshape(1, 64, 64, 1)
            pred = lip_model.predict(mouth_input)[0]
            lip_status = "Speaking 🗣️" if np.argmax(pred) == 1 else "Silent 🤫"
            lip_score = -0.5 if lip_status == "Speaking 🗣️" else 1

        # ---- Eye Detection ----
        left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        open_eyes = 0

        for eye_points in [left_eye, right_eye]:
            eye_roi = get_roi(frame, eye_points, w, h)
            if eye_roi.size == 0: continue
            eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            eye_resized = cv2.resize(eye_gray, (24, 24)) / 255.0
            eye_input = eye_resized.reshape(1, 24, 24, 1)
            pred = eye_model.predict(eye_input)[0][0]
            if pred > 0.5:
                open_eyes += 1

        eye_status = f"{open_eyes}/2 Eyes Open 👁️"
        eye_score = open_eyes / 2.0

        # ---- Head Movement Detection ----
        nose_points = [landmarks[i] for i in [1, 2, 4, 5, 98, 97, 195, 197]]
        nose_roi = get_roi(frame, nose_points, w, h)

        if nose_roi.size != 0:
            nose_gray = cv2.cvtColor(nose_roi, cv2.COLOR_BGR2GRAY)
            nose_resized = cv2.resize(nose_gray, (64, 64)) / 255.0
            nose_input = nose_resized.reshape(1, 64, 64, 1)
            pred = head_model.predict(nose_input)[0]
            label_idx = np.argmax(pred)
            head_status = LABELS[label_idx]
            head_score = 1 if head_status == "still" else -0.5

    # ---- Final Score ----
    final_score = max(0.0, round(((lip_score + eye_score + head_score) / 3) * 100, 2))

    # ---- Save to DB ----
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("attentiveness.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO full_attentiveness 
        (timestamp, lip_status, eye_status, head_status, lip_score, eye_score, head_score, final_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, lip_status, eye_status, head_status,
          lip_score, eye_score, head_score, final_score))
    conn.commit()
    conn.close()

    # ---- Display ----
    cv2.putText(frame, f"Lip: {lip_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"Eye: {eye_status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.putText(frame, f"Head: {head_status}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Attentiveness: {final_score}%", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Combined Attentiveness Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
