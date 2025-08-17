import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("head_movement.h5")

# Label mapping (adjust if needed)
LABELS = ["still", "moving"]

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
NOSE_LANDMARKS = [1, 2, 4, 5, 98, 97, 195, 197]

def extract_nose_roi(image, landmarks, img_w, img_h):
    nose_points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in NOSE_LANDMARKS]
    x_coords = [pt[0] for pt in nose_points]
    y_coords = [pt[1] for pt in nose_points]

    x_min = max(min(x_coords) - 10, 0)
    x_max = min(max(x_coords) + 10, img_w)
    y_min = max(min(y_coords) - 10, 0)
    y_max = min(max(y_coords) + 10, img_h)

    return image[y_min:y_max, x_min:x_max]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            nose_crop = extract_nose_roi(frame, face_landmarks.landmark, w, h)

            if nose_crop is not None and nose_crop.size != 0:
                gray = cv2.cvtColor(nose_crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                normalized = resized / 255.0
                input_data = np.expand_dims(normalized, axis=(0, -1))  # shape: (1, 64, 64, 1)

                # Predict
                prediction = model.predict(input_data)[0]
                label_index = np.argmax(prediction)
                label = LABELS[label_index]
                confidence = prediction[label_index]

                # Draw prediction
                cv2.putText(frame, f"{label} ({confidence:.2f})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Head Movement Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
