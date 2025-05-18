import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# ─── CONFIGURE PATHS ───────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")     # Expect: <project>/data/yes and /no
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")   # Will save here
os.makedirs(MODELS_DIR, exist_ok=True)

# Validate data directory
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Data folder not found at {DATA_DIR!r}. Make sure 'data/yes' and 'data/no' exist.")

categories = ["yes", "no"]
img_size   = 150

# ─── LOAD DATA ──────────────────────────────────────────────────────────────────
data = []  # holds [image_array, label]
for category in categories:
    folder_path = os.path.join(DATA_DIR, category)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Category folder missing: {folder_path!r}")
    label = 1 if category == "yes" else 0
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: unable to read {img_path}")
            continue
        img = cv2.resize(img, (img_size, img_size))
        data.append([img, label])

# Separate features and labels
X = np.array([item[0] for item in data]) / 255.0
y = np.array([item[1] for item in data])

# ─── TRAIN-TEST SPLIT ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── BUILD MODEL ────────────────────────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# ─── COMPILE & TRAIN ───────────────────────────────────────────────────────────
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=16)

# ─── SAVE MODEL ─────────────────────────────────────────────────────────────────
output_path = os.path.join(MODELS_DIR, "brain_tumor_model.h5")
model.save(output_path)
print(f"✅ Model saved to {output_path}")

# ─── EVALUATE ───────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.2f}")
