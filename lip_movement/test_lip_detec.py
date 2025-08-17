import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("lip_movement_model.h5")

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is in your directory

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    
    for face in faces:
        # Get landmarks
        landmarks = predictor(gray, face)
        
        # Get mouth region using landmark points (48 to 67)
        x_list = [landmarks.part(n).x for n in range(48, 68)]
        y_list = [landmarks.part(n).y for n in range(48, 68)]
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        # Add margin
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)

        # Crop mouth region
        mouth_roi = gray[y_min:y_max, x_min:x_max]
        if mouth_roi.size == 0:
            continue

        # Resize and normalize
        mouth_resized = cv2.resize(mouth_roi, (64, 64))
        mouth_normalized = mouth_resized / 255.0
        mouth_input = mouth_normalized.reshape(1, 64, 64, 1)

        # Predict
        prediction = model.predict(mouth_input)[0]
        label = "Speaking 🗣️" if np.argmax(prediction) == 1 else "Silent 🤫"

        # Show prediction on frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # Show frame
    cv2.imshow("Lip Movement Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
