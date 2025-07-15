import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import threading
import queue
import pyttsx3

# Configs
SEQUENCE_LENGTH = 40
INPUT_DIM = 126
LABELS = ['A','B','C','D','E','I','J']
CONFIDENCE_THRESHOLD = 0.95
model = load_model("sign_lstm_full.keras")

# TTS setup
tts_queue = queue.Queue()
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def tts_worker():
    while True:
        word = tts_queue.get()
        if word is None:
            break
        engine.say(word.strip())
        engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# MediaPipe and camera
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQUENCE_LENGTH)
predicted_text = ""
prev_letter = ""
frames_since_last_change = 0
min_gap_frames = 25
spoken_buffer = ""

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks]).flatten()

with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.75, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            hand = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0]
            if handedness.label == 'Right':
                norm = normalize_landmarks(hand.landmark)

                if len(sequence) > 0:
                    velocity = norm - sequence[-1][:63]
                else:
                    velocity = np.zeros_like(norm)

                combined = np.concatenate([norm, velocity])
                sequence.append(combined)

                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(np.array(sequence), axis=0)
                    prediction = model.predict(input_data, verbose=0)[0]
                    max_index = np.argmax(prediction)
                    confidence = prediction[max_index]
                    letter = LABELS[max_index]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        if letter == "Space":
                            if prev_letter != "Space" and not predicted_text.endswith(" "):
                                predicted_text += " "
                                prev_letter = "Space"
                                frames_since_last_change = 0
                                if spoken_buffer.strip():
                                    tts_queue.put(spoken_buffer)
                                    spoken_buffer = ""
                            else:
                                frames_since_last_change += 1
                        elif letter != prev_letter or frames_since_last_change > min_gap_frames:
                            predicted_text += letter
                            spoken_buffer += letter
                            prev_letter = letter
                            frames_since_last_change = 0
                        else:
                            frames_since_last_change += 1
        else:
            frames_since_last_change += 1

        cv2.putText(frame, f"Output: {predicted_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Real-time Sign-to-Text", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            predicted_text = ""
            spoken_buffer = ""
            prev_letter = ""
            frames_since_last_change = 0
            print("🧹 Text cleared")

# Cleanup
tts_queue.put(None)
tts_thread.join()
cap.release()
cv2.destroyAllWindows()
