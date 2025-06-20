import mediapipe as mp
import cv2
import csv
import os
from collections import defaultdict

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
output_file = "gesture_data.csv"
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize
label_counts = defaultdict(int)
buffer = []
buffer_limit = 50  # Write to file after every 50 samples

# Create file if not exists and load label counts
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']]
        header.append('label')
        writer.writerow(header)
else:
    with open(output_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 64:
                label_counts[row[-1]] += 1

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    normalized = []
    for lm in landmarks:
        normalized.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return normalized

current_label = None

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_image, (320, 240))
        results = hands.process(small_frame)

        # image = cv2.imread('a.png')
        # results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                score = handedness.classification[0].score
                label = handedness.classification[0].label  # 'Left' or 'Right'

                if score >= 0.8:  # Filter only confident hand detections
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    if current_label:
                        if label_counts[current_label] < 1000:
                            normalized = normalize_landmarks(hand_landmarks.landmark)
                            normalized.append(current_label)
                            buffer.append(normalized)
                            label_counts[current_label] += 1

                            if len(buffer) >= buffer_limit:
                                with open(output_file, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerows(buffer)
                                buffer.clear()
                        else:
                            print(f"✅ Label '{current_label}' reached 300 samples. Label cleared.")
                            current_label = None

        if current_label:
            count = label_counts[current_label]
            cv2.putText(frame, f"Label: {current_label} ({count}/300)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Landmarks: {len(hand_landmarks.landmark)}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Sign Language Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif 65 <= key <= 90 or 97 <= key <= 122:
            new_label = chr(key).upper()
            if label_counts[new_label] < 300:
                current_label = new_label
                print("Label set to:", current_label)
            else:
                print(f"⚠️  Label '{new_label}' already full.")
        elif key == ord('0'):
            current_label = None
            print("Label cleared.")

# Final flush to file
if buffer:
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(buffer)
    print(f"🗂️  Wrote remaining {len(buffer)} samples to file.")

capture.release()
cv2.destroyAllWindows()
