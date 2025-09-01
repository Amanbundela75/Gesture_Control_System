import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# --- Step 1: Dataset aur Parameters ko Set karna (Ismein koi badlaav nahi) ---

# Apne dataset ka path yahan daalein
DATA_PATH = "dataset/" 
# Model save karne ke liye naam
MODEL_NAME = "hand_gesture_model.h5"

# Sabhi images ke liye ek standard size
IMG_WIDTH, IMG_HEIGHT = 64, 64

# Training ke liye parameters
EPOCHS = 25
BATCH_SIZE = 32

# --- Step 2 & 3: Dataset Load, Preprocess aur Split karna (Naya aur Behtar Tareeka) ---

print("Dataset generator taiyaar kiya ja raha hai...")

# TensorFlow ke data generator ka istemal karein
# Yeh RAM mein saari images ek saath load nahi karega, isliye bade dataset ke liye best hai
image_size = (IMG_WIDTH, IMG_HEIGHT)
validation_split = 0.2 # 20% data testing/validation ke liye

# Training dataset generator
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=validation_split,
    subset="training",
    seed=123, # seed zaroori hai taaki training/validation data overlap na ho
    image_size=image_size,
    batch_size=BATCH_SIZE,
    color_mode='grayscale' # Humari images grayscale hain
)

# Validation (testing) dataset generator
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

# Gesture names (class names) ko dataset se lein
gesture_names = train_dataset.class_names
num_classes = len(gesture_names)
print(f"Total {num_classes} gestures found: {gesture_names}")

# Performance ke liye prefetching add karein
# Isse training ke dauraan data pipeline fast rehti hai
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- Step 4: CNN Model Banana (Ismein koi badlaav nahi) ---

print("CNN model banaya ja raha hai...")
model = Sequential([
    # Input layer mein humko pixel values ko 0-1 range mein laana hai
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    
    # 1st Convolutional Layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # 2nd Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # 3rd Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Data ko 1D array me convert karna
    Flatten(),
    
    # Overfitting se bachne ke liye Dropout layer
    Dropout(0.5),

    # Fully Connected Layer
    Dense(128, activation='relu'),
    
    # Output Layer
    Dense(num_classes, activation='softmax')
])

# Model ka summary print karein
model.summary()

# --- Step 5: Model ko Compile karna (Ismein koi badlaav nahi) ---

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Jab labels integer form mein hote hain
    metrics=['accuracy']
)

# --- Step 6: Model ko Train karna (Updated) ---

print("Model training shuru ho rahi hai...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    verbose=1
)

print("Model training poori ho gayi hai!")

# --- Step 7: Model ko Evaluate karna (Updated) ---

print("Model ko evaluate kiya ja raha hai...")
loss, accuracy = model.evaluate(validation_dataset, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- Step 8: Train kiye gaye Model ko Save karna (Ismein koi badlaav nahi) ---

model.save(MODEL_NAME)
print(f"Model '{MODEL_NAME}' naam se save ho gaya hai.")

# Gesture names ko bhi save kar sakte hain, taaki baad me predictions ke waqt kaam aaye
with open("gesture_labels.txt", "w") as f:
    f.write("\n".join(gesture_names))
print("Gesture labels 'gesture_labels.txt' file me save ho gaye hain.")