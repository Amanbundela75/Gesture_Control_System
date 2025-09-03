import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import time

# ==============================
# Global Setup
# ==============================
# 1. Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Load gesture labels
with open('gesture_labels.txt', 'r') as f:
    gesture_labels = f.read().splitlines()

# 3. Frame Dimensions & Other Variables
frame_w, frame_h = 640, 480
gesture_cooldown = 1.5
last_gesture_time = 0

# ==============================
# Main Loop
# ==============================
cap = cv2.VideoCapture(0)
cap.set(3, frame_w)
cap.set(4, frame_h)

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    # Define a Region of Interest (ROI) for prediction
    roi_x1, roi_y1, roi_x2, roi_y2 = 300, 100, 600, 400
    cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    # Preprocess the ROI for the model
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (64, 64))
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
    input_data = np.expand_dims(input_data, axis=-1)

    # Perform inference with the TFLite model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    gesture_index = np.argmax(prediction)
    predicted_gesture = gesture_labels[gesture_index]
    confidence = np.max(prediction)

    # Display the prediction on the screen
    cv2.putText(image, f"Prediction: {predicted_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Perform action based on high-confidence prediction
    current_time = time.time()
    if confidence > 0.95 and (current_time - last_gesture_time) > gesture_cooldown and "NoGesture" not in predicted_gesture:
        if "Screenshot" in predicted_gesture:
            pyautogui.screenshot(f"screenshot_{int(time.time())}.png")
            print("Action: Screenshot taken")
            last_gesture_time = current_time
        # Add other functions here based on your trained gestures
        # Note: Pointer and Drawing modes work better with the rule-based approach

    cv2.imshow("AI Gesture Control", image)

    if cv2.waitKey(5) & 0xFF == 27: # 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()