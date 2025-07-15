import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

# === CONFIGURATION ===
LABEL = "J"
VARIATION = "fast"
DATASET_PATH = "sequence_dataset"
NUM_SAMPLES = 10                
SEQUENCE_LENGTH = 40
FPS = 20

def load_all_indices(label, variation):
    folder = os.path.join(DATASET_PATH, label, variation)
    files = [f for f in os.listdir(folder) if f.endswith(".npz") and "seq_" in f]
    indices = [int(f.split("_")[1].split(".")[0]) for f in files]
    original_indices = [i for i in indices if i % 3 == 0]
    return sorted(original_indices)

def load_sequence(label, variation, idx):
    path = os.path.join(DATASET_PATH, label, variation, f"seq_{idx}.npz")
    return np.load(path)["sequence"]

def plot_hand_frame(ax, frame):
    ax.clear()
    landmarks = frame[:63].reshape(-1, 3)
    xs, ys = landmarks[:, 0], -landmarks[:, 1]
    ax.scatter(xs, ys, c='blue', s=10)

    connections = [(0, 1), (1, 2), (2, 3), (3, 4),
                   (0, 5), (5, 6), (6, 7), (7, 8),
                   (5, 9), (9,10), (10,11), (11,12),
                   (9,13), (13,14), (14,15), (15,16),
                   (13,17), (17,18), (18,19), (19,20)]
    for i, j in connections:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], 'gray', linewidth=0.5)

    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.axis('off')

# === DYNAMIC SETUP ===
available_indices = load_all_indices(LABEL, VARIATION)
selected_indices = random.sample(available_indices, min(NUM_SAMPLES, len(available_indices)))
sequences = [load_sequence(LABEL, VARIATION, idx) for idx in selected_indices]

# Calculate rows and columns dynamically
COLS = math.ceil(NUM_SAMPLES**0.5)
ROWS = math.ceil(NUM_SAMPLES / COLS)

# Create subplots
fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))
axes = np.array(axes).flatten()

# Pad axes if more than needed
for i in range(NUM_SAMPLES, len(axes)):
    axes[i].axis('off')

def update(frame_idx):
    for i, seq in enumerate(sequences):
        if frame_idx < len(seq):
            plot_hand_frame(axes[i], seq[frame_idx])
            axes[i].set_title(f"Seq {selected_indices[i]}")
    return axes

ani = animation.FuncAnimation(fig, update, frames=SEQUENCE_LENGTH, interval=1000//FPS, blit=False, repeat=True)

plt.tight_layout()
plt.show()
