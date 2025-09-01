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
# ==============================
# Global Setup
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 3. Pointer Control ke liye Variables
cap = cv2.VideoCapture(0)

gesture_cooldown = 1.5
last_gesture_time = 0
initial_wrist_x = None
swipe_threshold = 0.15
tip_ids = [4, 8, 12, 16, 20]
screen_w, screen_h = pyautogui.size()
frame_w, frame_h = 640, 480 # Webcam frame ka size
smoothing = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

drawing_mode_on = False  # 🔹 Drawing mode state

# --- Main Loop ---
cap = cv2.VideoCapture(0)
cap.set(3, frame_w)
cap.set(4, frame_h)

while True:

# ==============================
# Helper Functions
# ==============================

def count_fingers(landmarks):
    """Return finger states [1/0] and total fingers up"""
    fingers = []

    # Thumb
    if landmarks[tip_ids[0]][1] > landmarks[tip_ids[0] - 2][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for id in range(1, 5):
        if landmarks[tip_ids[id]][2] < landmarks[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers, fingers.count(1)


def pointer_mode(landmarks, image, w, h):
    """Move mouse with index finger"""
    cv2.putText(image, 'Pointer Mode', (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    index_x, index_y = landmarks[8][1], landmarks[8][2]
    cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), cv2.FILLED)

    screen_x = np.interp(index_x, [w * 0.2, w * 0.8], [0, screen_w])
    screen_y = np.interp(index_y, [h * 0.2, h * 0.8], [0, screen_h])
    pyautogui.moveTo(screen_x, screen_y)


def swipe_gesture(my_hand, current_time):
    """Handle left/right swipe"""
    global initial_wrist_x, last_gesture_time
    wrist_landmark = my_hand.landmark[mp_hands.HandLandmark.WRIST]

    if initial_wrist_x is None:
        initial_wrist_x = wrist_landmark.x
    else:
        delta_x = wrist_landmark.x - initial_wrist_x
        if delta_x > swipe_threshold:
            pyautogui.press('right')
            print("Right Swipe")
            last_gesture_time = current_time
            initial_wrist_x = None
        elif delta_x < -swipe_threshold:
            pyautogui.press('left')
            print("Left Swipe")
            last_gesture_time = current_time
            initial_wrist_x = None


def volume_control(landmarks, current_time, image):
    """Volume up/down based on hand size"""
    global last_gesture_time
    wrist = landmarks[0][1], landmarks[0][2]
    mcp_middle = landmarks[9][1], landmarks[9][2]
    hand_size = math.hypot(mcp_middle[0] - wrist[0], mcp_middle[1] - wrist[1])

    if hand_size > 150:
        pyautogui.press('volumedown')
        print("Volume Down")
        last_gesture_time = current_time
    elif hand_size < 80:
        pyautogui.press('volumeup')
        print("Volume Up")
        last_gesture_time = current_time

    cv2.putText(image, 'Volume Mode', (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


def take_screenshot(current_time, image):
    """Take screenshot"""
    global last_gesture_time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    pyautogui.screenshot(filename)
    print(f"Screenshot saved as {filename}")
    cv2.putText(image, 'Screenshot Taken!', (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    last_gesture_time = current_time


def drawing_mode(landmarks, w, h, image):
    """Drawing mode with index + middle finger"""
    global drawing_mode_on
    cv2.putText(image, 'Drawing Mode', (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    index_tip = landmarks[8][1], landmarks[8][2]
    middle_tip = landmarks[12][1], landmarks[12][2]

    # Mouse follows index finger
    screen_x = np.interp(index_tip[0], [w * 0.2, w * 0.8], [0, screen_w])
    screen_y = np.interp(index_tip[1], [h * 0.2, h * 0.8], [0, screen_h])
    pyautogui.moveTo(screen_x, screen_y)

    # Distance between index and middle finger
    distance = math.hypot(middle_tip[0] - index_tip[0], middle_tip[1] - index_tip[1])

    if distance < 40:  # Fingers close → Draw
        if not drawing_mode_on:
            pyautogui.mouseDown()
            drawing_mode_on = True
            print("Mouse Down - Drawing Started")
        cv2.circle(image, index_tip, 10, (0, 0, 255), cv2.FILLED)
    else:  # Fingers apart → Stop Drawing
        if drawing_mode_on:
            pyautogui.mouseUp()
            drawing_mode_on = False
            print("Mouse Up - Drawing Stopped")


# ==============================
# Main Loop
# ==============================
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image = cv2.flip(image, 1)
    
    # MediaPipe ke liye image ko process karein
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    action_text = "No Action"

        continue

    h, w, c = image.shape
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
        my_hand = results.multi_hand_landmarks[0]
        landmark_list = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(my_hand.landmark)]
        mp_draw.draw_landmarks(image, my_hand, mp_hands.HAND_CONNECTIONS)

        if landmark_list:
            fingers, total_fingers = count_fingers(landmark_list)
            cv2.putText(image, f'Fingers: {total_fingers}', (10, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            current_time = time.time()
            if current_time - last_gesture_time > gesture_cooldown:

                if total_fingers == 1 and fingers[1] == 1:
                    pointer_mode(landmark_list, image, w, h)

                elif total_fingers == 5:
                    swipe_gesture(my_hand, current_time)

                elif total_fingers == 0:
                    volume_control(landmark_list, current_time, image)

                elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                    take_screenshot(current_time, image)

            # Drawing mode (always checked)
            if total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                drawing_mode(landmark_list, w, h, image)

    cv2.imshow("Gesture Control Super Project", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
