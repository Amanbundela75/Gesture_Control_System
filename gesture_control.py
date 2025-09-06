import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import mediapipe as mp
import time

# --- Setup ---
# 1. TFLite Model Load Karein
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
with open('gesture_labels.txt', 'r') as f:
    gesture_labels = f.read().splitlines()

# 2. MediaPipe Hands Setup Karein
mp_hands = mp.solutions.hands
# === YAHAN FIX KIYA GAYA HAI ===
# mp.hands.Hands ke bajaye mp_hands.Hands ka istemal karein
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# 3. Baaki Variables
screen_w, screen_h = pyautogui.size()
frame_w, frame_h = 640, 480
initial_wrist_x = None
swipe_threshold = 0.15
last_swipe_time = 0

# --- Main Loop ---
cap = cv2.VideoCapture(0)
cap.set(3, frame_w)
cap.set(4, frame_h)

while True:
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    action_text = ""

    if results.multi_hand_landmarks:
        my_hand = results.multi_hand_landmarks[0]
        
        # Landmark coordinates lein
        lmList = []
        for id, lm in enumerate(my_hand.landmark):
            lmList.append([id, int(lm.x * frame_w), int(lm.y * frame_h)])
        
        # Ungliyan ginein
        fingers_up = []
        tipIds = [4, 8, 12, 16, 20]
        # Thumb
        if lmList and len(lmList) > tipIds[0]: # Check if landmarks are detected
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]: fingers_up.append(1)
            else: fingers_up.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]: fingers_up.append(1)
                else: fingers_up.append(0)
        
        totalFingers = fingers_up.count(1)
        
        # === Decision Logic (Hybrid System) ===
        # 1. Pointer Mode (Rule-based)
        if totalFingers == 1 and fingers_up[1] == 1:
            action_text = "Pointer Mode"
            x, y = lmList[8][1], lmList[8][2] # Index finger tip
            screen_x = np.interp(x, (100, frame_w - 100), (0, screen_w))
            screen_y = np.interp(y, (100, frame_h - 100), (0, screen_h))
            pyautogui.moveTo(screen_x, screen_y)
        
        # 2. Swipe Gesture (Rule-based)
        elif totalFingers == 5:
            wrist_x = my_hand.landmark[mp_hands.HandLandmark.WRIST].x
            if initial_wrist_x is None:
                initial_wrist_x = wrist_x
                last_swipe_time = time.time()
            elif time.time() - last_swipe_time > 0.5: # Cooldown
                delta_x = wrist_x - initial_wrist_x
                if delta_x > swipe_threshold:
                    pyautogui.press('right')
                    action_text = "Swipe Right"
                    initial_wrist_x = None
                elif delta_x < -swipe_threshold:
                    pyautogui.press('left')
                    action_text = "Swipe Left"
                    initial_wrist_x = None
        
        # 3. AI Model Prediction (For other cases)
        else:
            initial_wrist_x = None # Reset swipe tracking
            roi = image[100:400, 300:600]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            input_data = np.expand_dims(resized, axis=0).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=-1)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            gesture = gesture_labels[np.argmax(prediction)]
            action_text = f"Prediction: {gesture}"

    # Text aur image display karein
    cv2.rectangle(image, (300, 100), (600, 400), (0, 255, 0), 2)
    cv2.putText(image, action_text, (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Gesture Control", image)

    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()