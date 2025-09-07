import cv2
import numpy as np
import tensorflow as tf
import os

# --- Setup ---
# 1. Model aur Labels Load Karein
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
with open('gesture_labels.txt', 'r') as f:
    gesture_labels = f.read().splitlines()

# 2. Test Dataset ka Path Set Karein
# Apne screenshot ke anusaar, test folders is path ke andar hain
TEST_DATA_PATH = "dataset/asl_alphabet_test/asl_alphabet_test/"

# --- Evaluation Logic ---
total_images = 0
correct_predictions = 0

print("Starting evaluation on the test dataset...")

# Test folder ke har gesture folder mein jaayein
for gesture_folder in os.listdir(TEST_DATA_PATH):
    gesture_path = os.path.join(TEST_DATA_PATH, gesture_folder)
    
    if not os.path.isdir(gesture_path):
        continue

    # Har folder ke andar ki images ko test karein
    for image_file in os.listdir(gesture_path):
        image_path = os.path.join(gesture_path, image_file)
        
        try:
            # Image ko load aur preprocess karein
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (64, 64))
            
            input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=-1)
            
            # Prediction karein
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_gesture = gesture_labels[np.argmax(prediction)]
            true_gesture = gesture_folder
            
            # Result check karein
            if predicted_gesture == true_gesture:
                correct_predictions += 1
            total_images += 1
        except Exception as e:
            print(f"Could not process image {image_file}: {e}")

print("Evaluation complete.")

# Final Accuracy Calculate aur Print Karein
if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100
    print("\n--- Evaluation Result ---")
    print(f"Total Images Tested: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Model Accuracy on Test Set: {accuracy:.2f}%")
else:
    print("No images found in the test directory.")