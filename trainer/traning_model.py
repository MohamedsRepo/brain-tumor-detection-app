import os, cv2, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# ✅ Use data inside .venv/data/yes and .venv/data/no
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, ".venv", "data")
categories = ["yes", "no"]
img_size = 150

data = []

# ✅ Load and label the data (1 = tumor, 0 = no tumor)
for category in categories:
    path = os.path.join(data_dir, category)
    label = 1 if category == "yes" else 0

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Skipped unreadable file: {img_name}")
                continue
            img = cv2.resize(img, (img_size, img_size))
            data.append([img, label])
        except Exception as e:
            print(f"❌ Failed to load {img_name}: {e}")

# ✅ Shuffle, normalize, and split
np.random.shuffle(data)
X, y = zip(*data)
X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# ✅ Compile & train
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=16)

# ✅ Save model
os.makedirs("models", exist_ok=True)
model.save("models/brain_tumor_model.h5")

# ✅ Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.2f}")
