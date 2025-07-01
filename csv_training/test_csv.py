import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

# Load trained model
model = load_model("sign_model_AtoF_final.keras")

# Label encoder (optional — manually define A–P)
labels = [chr(i) for i in range(ord('A'), ord('P')+1)]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Normalize relative to wrist
def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    norm = []
    for lm in landmarks:
        norm.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return norm

# Webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
) as hands:
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract and normalize
                    norm = normalize_landmarks(hand_landmarks.landmark)
                    if len(norm) == 63:
                        input_data = np.array(norm).reshape(1, -1)

                        # Same normalization as training
                        input_data = (input_data - input_data.mean(axis=1, keepdims=True)) / input_data.std(axis=1, keepdims=True)

                        pred = model.predict(input_data)
                        pred_label = labels[np.argmax(pred)]

                        # Show prediction
                        cv2.putText(frame, f"Prediction: {pred_label}", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Sign Language Prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        import traceback
        traceback.print_exc()

cap.release()
cv2.destroyAllWindows()
