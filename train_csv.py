import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Load Data
df = pd.read_csv('gesture_data.csv')
X = df.iloc[:, :-1].values  # 63 landmarks
y = df.iloc[:, -1].values   # labels A–F

# Normalize landmarks (zero mean, unit variance)
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1  # Prevent divide by zero
X = (X - mean) / std

# Encode A–F → 0–5
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=16)

# Stratified Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

# Compute class weights for imbalance
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = dict(enumerate(class_weights))

# Optional: Add small noise to training data for better generalization
def add_noise(data, level=0.01):
    return data + level * np.random.randn(*data.shape)

X_train = add_noise(X_train)

# Build the model
model = Sequential([
    Dense(256, input_shape=(63,)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),

    Dense(16, activation='softmax')  # Output for A–P
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ModelCheckpoint("best_model_AtoF.keras", save_best_only=True)
]

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=32,
                    class_weight=class_weights_dict,
                    callbacks=callbacks)

# Save final model
model.save("sign_model_AtoF_final.keras")
print("✅ Model saved as 'sign_model_AtoF_final.keras'")

# Evaluate and visualize confusion matrix
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
