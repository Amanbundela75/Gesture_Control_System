import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling

# Parameters
DATA_PATH = "dataset/" 
MODEL_NAME = "simple_gesture_model.h5"
IMG_WIDTH, IMG_HEIGHT = 64, 64
EPOCHS = 15
BATCH_SIZE = 32

# Load dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, color_mode='grayscale'
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, color_mode='grayscale'
)

gesture_names = train_dataset.class_names
num_classes = len(gesture_names)
print(f"Gestures found: {gesture_names}")

# Build Model
model = Sequential([
    Rescaling(1./255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    Conv2D(32, (3, 3), activation='relu'), MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile and Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)

# === YAHAN CODE ADD KIYA GAYA HAI ===
# Final evaluation on the validation set
print("\n--- Final Evaluation ---")
loss, accuracy = model.evaluate(validation_dataset, verbose=0)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

# Save
model.save(MODEL_NAME)
with open("gesture_labels.txt", "w") as f:
    f.write("\n".join(gesture_names))
print(f"\nModel saved as {MODEL_NAME} and labels saved to gesture_labels.txt")