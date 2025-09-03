import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling, RandomFlip, RandomRotation, RandomZoom

# --- Step 1: Dataset aur Parameters ko Set karna ---

# === YAHAN BADLAAV KIYA GAYA HAI ===
# Path ko update karke 'asl_alphabet_train' folder point kiya gaya hai
DATA_PATH = "dataset/asl_alphabet_train/" 

MODEL_NAME = "hand_gesture_model_asl.h5" # Model ka naya naam
IMG_WIDTH, IMG_HEIGHT = 64, 64
EPOCHS = 30
BATCH_SIZE = 32

# --- Step 2 & 3: Dataset Load, Preprocess aur Split karna ---
print("Dataset generator taiyaar kiya ja raha hai...")
image_size = (IMG_WIDTH, IMG_HEIGHT)
validation_split = 0.2 # Using 20% of the data for validation

# Check if the directory exists and is not empty
if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    print(f"Error: The directory '{DATA_PATH}' is empty or does not exist.")
    print("Please make sure your dataset is correctly placed.")
    exit()

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

gesture_names = train_dataset.class_names
num_classes = len(gesture_names)
print(f"Total {num_classes} gestures found: {gesture_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- Step 4: CNN Model Banana (Data Augmentation ke Saath) ---
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.2),
])

print("CNN model with Data Augmentation banaya ja raha hai...")
model = Sequential([
    Rescaling(1./255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    data_augmentation,
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# --- Step 5: Model ko Compile karna ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Step 6: Model ko Train karna ---
print("Model training shuru ho rahi hai...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    verbose=1
)

# --- Step 7 & 8: Model ko Evaluate aur Save karna ---
loss, accuracy = model.evaluate(validation_dataset, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

model.save(MODEL_NAME)
print(f"Model '{MODEL_NAME}' naam se save ho gaya hai.")

with open("gesture_labels.txt", "w") as f:
    f.write("\n".join(gesture_names))
print("Gesture labels 'gesture_labels.txt' file me save ho gaye hain.")