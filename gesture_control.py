import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import mediapipe as mp # MediaPipe ko import karein

# --- Setup ---

# 1. AI Model aur Labels Load Karna
model = tf.keras.models.load_model('hand_gesture_model.h5')
with open('gesture_labels.txt', 'r') as f:
    gesture_labels = f.read().splitlines()

# 2. MediaPipe Hands Setup Karna
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 3. Pointer Control ke liye Variables
screen_w, screen_h = pyautogui.size()
frame_w, frame_h = 640, 480 # Webcam frame ka size
smoothing = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# --- Main Loop ---
cap = cv2.VideoCapture(0)
cap.set(3, frame_w)
cap.set(4, frame_h)

while True:
    success, image = cap.read()
    if not success:
        break
    
    image = cv2.flip(image, 1)
    
    # MediaPipe ke liye image ko process karein
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    action_text = "No Action"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Ungliyon ke tips ke coordinates lein
        lmList = []
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
        
        # Check karein kaun si ungliyan upar hain
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalFingers = fingers.count(1)

        # === Feature 1: Pointer Control ===
        if totalFingers == 1 and fingers[1] == 1: # Agar sirf Index finger upar hai
            action_text = "Pointer Mode"
            # Index finger ke tip (landmark 8) ka coordinate lein
            x1, y1 = lmList[8][1], lmList[8][2]
            
            # Screen coordinates mein convert karein
            x3 = np.interp(x1, (100, frame_w - 100), (0, screen_w))
            y3 = np.interp(y1, (100, frame_h - 100), (0, screen_h))
            
            # Pointer ko smooth karein
            clocX = plocX + (x3 - plocX) / smoothing
            clocY = plocY + (y3 - plocY) / smoothing
            
            pyautogui.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

        # === Feature 2: AI Model se Static Gesture Prediction ===
        else:
            # ROI (Region of Interest) se prediction karein
            roi_x1, roi_y1, roi_x2, roi_y2 = 300, 100, 600, 400
            cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Image ko model ke liye preprocess karein
            gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (64, 64))
            input_image = resized_image.reshape(1, 64, 64, 1)

            # Prediction
            prediction = model.predict(input_image, verbose=0)
            gesture_index = np.argmax(prediction)
            predicted_gesture = gesture_labels[gesture_index]
            confidence = np.max(prediction)

            # Action text set karein aur action perform karein
            if confidence > 0.9:
                if predicted_gesture == 'Peace':
                    action_text = "Peace: Taking Screenshot"
                    pyautogui.screenshot(f'screenshot_{pyautogui.screenshot().size[0]}.png') # Quick save
                elif predicted_gesture == 'Fist':
                    action_text = "Fist: Paused"
                    # Yahan aap koi aur action daal sakte hain
                elif predicted_gesture == 'OpenPalm':
                    action_text = "OpenPalm: Ready"
                else:
                    action_text = predicted_gesture

    # Screen par text dikhayein
    cv2.putText(image, action_text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Gesture Control", image)

    if cv2.waitKey(5) & 0xFF == 27: # Esc key se band karein
        break

cap.release()
cv2.destroyAllWindows()