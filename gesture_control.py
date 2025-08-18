import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# --- सेटअप ---
mp_solutions = mp.solutions
mp_hands = mp_solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp_solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- जेस्चर कंट्रोल वेरिएबल्स ---
gesture_cooldown = 1.5 # स्क्रीनशॉट के लिए कूलडाउन बढ़ाया
last_gesture_time = 0
initial_wrist_x = None
swipe_threshold = 0.15
tip_ids = [4, 8, 12, 16, 20]
screen_w, screen_h = pyautogui.size()

print("प्रोग्राम शुरू हो रहा है... Esc दबाकर बाहर निकलें।")

# --- मुख्य लूप ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    h, w, c = image.shape
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    landmark_list = []
    if results.multi_hand_landmarks:
        my_hand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(my_hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append([id, cx, cy])
        
        mp_draw.draw_landmarks(image, my_hand, mp_hands.HAND_CONNECTIONS)

        if len(landmark_list) != 0:
            fingers = []
            if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            total_fingers = fingers.count(1)
            cv2.putText(image, f'Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            current_time = time.time()
            if current_time - last_gesture_time > gesture_cooldown:

                # ===== फीचर 1: लेजर पॉइंटर मोड (1 उंगली) =====
                if total_fingers == 1 and fingers[1] == 1:
                    cv2.putText(image, 'Pointer Mode', (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    index_x, index_y = landmark_list[8][1], landmark_list[8][2]
                    cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), cv2.FILLED)
                    screen_x = np.interp(index_x, [w*0.2, w*0.8], [0, screen_w])
                    screen_y = np.interp(index_y, [h*0.2, h*0.8], [0, screen_h])
                    pyautogui.moveTo(screen_x, screen_y)

                # ===== फीचर 2: स्वाइप जेस्चर (5 उंगलियां) =====
                elif total_fingers == 5:
                    wrist_landmark = my_hand.landmark[mp_hands.HandLandmark.WRIST]
                    if initial_wrist_x is None:
                        initial_wrist_x = wrist_landmark.x
                    else:
                        delta_x = wrist_landmark.x - initial_wrist_x
                        if delta_x > swipe_threshold:
                            pyautogui.press('right'); print("Right Swipe")
                            last_gesture_time = current_time; initial_wrist_x = None
                        elif delta_x < -swipe_threshold:
                            pyautogui.press('left'); print("Left Swipe")
                            last_gesture_time = current_time; initial_wrist_x = None

                # ===== फीचर 3: वॉल्यूम कंट्रोल (मुट्ठी) =====
                elif total_fingers == 0:
                    wrist = landmark_list[0][1], landmark_list[0][2]
                    mcp_middle = landmark_list[9][1], landmark_list[9][2]
                    hand_size = math.hypot(mcp_middle[0] - wrist[0], mcp_middle[1] - wrist[1])
                    if hand_size > 150:
                        pyautogui.press('volumedown'); print("Volume Down")
                        last_gesture_time = current_time
                    elif hand_size < 80:
                        pyautogui.press('volumeup'); print("Volume Up")
                        last_gesture_time = current_time
                    cv2.putText(image, 'Volume Mode', (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


                # ===== नया फीचर 4: स्क्रीनशॉट (2 उंगलियां) =====
                elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                    # एक यूनिक फाइलनाम बनाने के लिए टाइमस्टैम्प का उपयोग करें
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                    pyautogui.screenshot(filename)
                    print(f"Screenshot saved as {filename}")
                    cv2.putText(image, 'Screenshot Taken!', (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    last_gesture_time = current_time
            else:
                pass

    cv2.imshow("Gesture Control Super Project", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()