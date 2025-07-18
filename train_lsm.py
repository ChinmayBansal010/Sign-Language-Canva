import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import Input


# Constants
DATASET_PATH = "sequence_dataset"
SEQUENCE_LENGTH = 40
INPUT_DIM = 126  # 63 normalized + 63 velocity
VARIATIONS = ["normal", "fast", "slow", "tilted"]
LABELS = ['A', 'B', 'C', 'D', 'E', 'I', 'J']
label_map = {label: i for i, label in enumerate(LABELS)}

# Step 1: Load all data
X, y = [], []

for label in LABELS:
    for variation in VARIATIONS:
        folder = os.path.join(DATASET_PATH, label, variation)
        for file in os.listdir(folder):
            if file.endswith(".npz"):
                data = np.load(os.path.join(folder, file))["sequence"]
                if data.shape == (SEQUENCE_LENGTH, INPUT_DIM):
                    X.append(data)
                    y.append(label_map[label])

X = np.array(X)
y = to_categorical(y, num_classes=len(LABELS))

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Step 3: Build model
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, INPUT_DIM)),
    LSTM(128, return_sequences=True),
    Dropout(0.4),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(LABELS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=32)

# Step 5: Save model
model.save("sign_lstm_full.keras")

# Step 6: Confusion Matrix
# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)

# Plot
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()
