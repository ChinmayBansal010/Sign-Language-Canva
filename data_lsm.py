import cv2
import os
import numpy as np
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

DATASET_PATH = "sequence_dataset"
SEQUENCE_LENGTH = 40
MAX_PER_VARIATION = 75
AUG_PER_SEQUENCE = 2
TARGET_FPS = 10
MAX_ORIGINAL_CAPTURES = MAX_PER_VARIATION // (1 + AUG_PER_SEQUENCE)

LABELS = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)] + ["Space"]
VARIATIONS = ["normal", "fast", "slow", "tilted"]

executor = ThreadPoolExecutor()

for label in LABELS:
    for variation in VARIATIONS:
        os.makedirs(os.path.join(DATASET_PATH, label, variation), exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    return np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks]).flatten()

def compute_velocity(prev_frame, curr_frame):
    return curr_frame - prev_frame

def augment_sequence(seq):
    augmented = []
    for _ in range(AUG_PER_SEQUENCE):
        scale = 1 + np.random.uniform(-0.02, 0.02)
        noise = np.random.normal(0, 0.01, seq.shape)
        augmented.append(seq * scale)
        augmented.append(seq + noise)
    return augmented[:AUG_PER_SEQUENCE]

def load_existing_sequences():
    counts, cache = {}, {}
    for label in LABELS:
        for variation in VARIATIONS:
            folder = os.path.join(DATASET_PATH, label, variation)
            key = (label, variation)
            cache[key] = []
            counts[key] = 0
            if not os.path.exists(folder):
                continue
            files = [f for f in os.listdir(folder) if f.endswith('.npz')]
            for f in files:
                try:
                    arr = np.load(os.path.join(folder, f))['sequence']
                    cache[key].append(arr)
                except:
                    continue
            counts[key] = len(cache[key]) // (1 + AUG_PER_SEQUENCE)
    return counts, cache

def save_sequence_async(seq, label, variation, idx):
    print(f"SAVING: {label}/{variation}/seq_{idx}.npz")
    folder = os.path.join(DATASET_PATH, label, variation)
    filename = f"seq_{idx}.npz"
    path = os.path.join(folder, filename)
    np.savez_compressed(path, sequence=seq)

def beep():
    print("\a")

current_label, variation_type = None, "normal"
sequence = []
recording_enabled = True
variation_counts, saved_sequences_cache = load_existing_sequences()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_frame_time = time.time()

with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        time.sleep(max(0, (1 / TARGET_FPS) - (time.time() - last_frame_time)))
        last_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0]

            if handedness.label == 'Right' and handedness.score >= 0.8:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if current_label and recording_enabled:
                    normalized = normalize_landmarks(hand_landmarks.landmark)

                    if sequence:
                        velocity = compute_velocity(sequence[-1][:63], normalized)
                        combined = np.concatenate([normalized, velocity])
                    else:
                        combined = np.concatenate([normalized, np.zeros_like(normalized)])

                    sequence.append(combined)

                    if len(sequence) == SEQUENCE_LENGTH:
                        key = (current_label, variation_type)
                        current_sequence = np.array(sequence)

                        if variation_counts[key] < MAX_ORIGINAL_CAPTURES:
                            idx = variation_counts[key] * (1 + AUG_PER_SEQUENCE)
                            executor.submit(save_sequence_async, current_sequence, current_label, variation_type, idx)
                            saved_sequences_cache[key].append(current_sequence)

                            for i, aug_seq in enumerate(augment_sequence(current_sequence), start=1):
                                executor.submit(save_sequence_async, aug_seq, current_label, variation_type, idx + i)
                                saved_sequences_cache[key].append(aug_seq)

                            variation_counts[key] += 1
                            print(f"✅ Saved: {key} | Original count: {variation_counts[key]}")

                            if variation_counts[key] >= MAX_ORIGINAL_CAPTURES:
                                recording_enabled = False
                                beep()
                                print(f"🛑 {variation_type.upper()} DONE for {current_label}")

                        sequence = []

        if current_label:
            count = variation_counts.get((current_label, variation_type), 0) * (1 + AUG_PER_SEQUENCE)
            cv2.putText(frame, f"Label: {current_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Mode: {variation_type} ({count}/{MAX_PER_VARIATION})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if not recording_enabled:
                cv2.putText(frame, f"{variation_type.upper()} DONE!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"SeqLen: {len(sequence)}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Sequence Collector", frame)

        key = cv2.waitKeyEx(1) 
        if key == ord('q'):
            break
        elif key == ord(' '):
            current_label = "Space"
        elif 65 <= key <= 90 or 97 <= key <= 122:
            current_label = chr(key).upper()
        elif key in range(ord('0'), ord('9') + 1):
            current_label = chr(key)
        elif key == ord('0'):
            current_label = None
            recording_enabled = False
            sequence = []
            print("❌ Label cleared")
        elif key in [ord('['), ord(']'), ord(';'), ord('\'')]:
            key_to_variation = {
                ord('['): "normal",
                ord(']'): "fast",
                ord(';'): "slow",
                ord('\''): "tilted"
            }
            variation_type = key_to_variation[key]
            if current_label and variation_counts.get((current_label, variation_type), 0) < MAX_ORIGINAL_CAPTURES:
                recording_enabled = True
            else:
                recording_enabled = False
            sequence = []
            print(f"🔁 Switched to: {variation_type}")

        if current_label and not recording_enabled and variation_counts.get((current_label, variation_type), 0) < MAX_ORIGINAL_CAPTURES:
            recording_enabled = True
            sequence = []
            print("🔴 Recording started for:", current_label)

        total_done = all(variation_counts[(lbl, var)] >= MAX_ORIGINAL_CAPTURES for lbl in LABELS for var in VARIATIONS)
        if total_done:
            print("🎉 All variations for all labels completed.")
            break

cap.release()
cv2.destroyAllWindows()
