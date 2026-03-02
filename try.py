import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Get coordinates
            nose = face_landmarks.landmark[NOSE_TIP]
            left_cheek = face_landmarks.landmark[LEFT_CHEEK]
            right_cheek = face_landmarks.landmark[RIGHT_CHEEK]

            nose_x = int(nose.x * w)
            left_x = int(left_cheek.x * w)
            right_x = int(right_cheek.x * w)

            # Draw points
            cv2.circle(frame, (nose_x, int(nose.y * h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (left_x, int(left_cheek.y * h)), 5, (255, 0, 0), -1)
            cv2.circle(frame, (right_x, int(right_cheek.y * h)), 5, (0, 0, 255), -1)

            # Compare distances
            dist_left = abs(nose_x - left_x)
            dist_right = abs(right_x - nose_x)

            if dist_left < dist_right - 10:
                direction = "Looking Right"
            elif dist_right < dist_left - 10:
                direction = "Looking Left"
            else:
                direction = "Looking Forward"

            cv2.putText(frame, direction, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Head Direction Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()