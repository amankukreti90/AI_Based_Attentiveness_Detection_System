import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("eye_state_model.h5")

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (ex, ey, ew, eh) in eyes:
        eye_roi = gray[ey:ey+eh, ex:ex+ew]
        eye_resized = cv2.resize(eye_roi, (24, 24))
        eye_normalized = eye_resized / 255.0
        eye_input = eye_normalized.reshape(1, 24, 24, 1)

        prediction = model.predict(eye_input)[0][0]
        label = "Open 👁️" if prediction > 0.5 else "Closed 😴"

        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.putText(frame, label, (ex, ey - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Eye State Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model

# # Load your trained model
# model = load_model("eye_state_model.h5")

# # Initialize mediapipe face mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# # Define eye landmark indices for right and left eye
# RIGHT_EYE = [33, 133]
# LEFT_EYE = [362, 263]

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = face_mesh.process(rgb_frame)

#     if result.multi_face_landmarks:
#         for face_landmarks in result.multi_face_landmarks:
#             for eye_indices, label in zip([RIGHT_EYE, LEFT_EYE], ["R", "L"]):
#                 x1 = int(face_landmarks.landmark[eye_indices[0]].x * w)
#                 y1 = int(face_landmarks.landmark[eye_indices[0]].y * h)
#                 x2 = int(face_landmarks.landmark[eye_indices[1]].x * w)
#                 y2 = int(face_landmarks.landmark[eye_indices[1]].y * h)

#                 # Define a region around the eye
#                 x_min = min(x1, x2) - 10
#                 y_min = min(y1, y2) - 10
#                 x_max = max(x1, x2) + 10
#                 y_max = max(y1, y2) + 10

#                 eye_roi = frame[y_min:y_max, x_min:x_max]
#                 if eye_roi.size == 0:
#                     continue

#                 gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
#                 resized_eye = cv2.resize(gray_eye, (24, 24))
#                 normalized_eye = resized_eye / 255.0
#                 input_eye = normalized_eye.reshape(1, 24, 24, 1)

#                 prediction = model.predict(input_eye)[0][0]
#                 state = "Open 👁️" if prediction > 0.5 else "Closed 😴"

#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label} Eye: {state}", (x_min, y_min - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

#     cv2.imshow("Eye State Detection (MediaPipe)", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
