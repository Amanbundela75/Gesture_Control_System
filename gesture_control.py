import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np # Numpy को इम्पोर्ट करें

# --- सेटअप ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- जेस्चर कंट्रोल वेरिएबल्स ---
gesture_cooldown = 1.5
last_gesture_time = 0
initial_wrist_x = None
swipe_threshold = 0.15
tip_ids = [4, 8, 12, 16, 20]
screen_w, screen_h = pyautogui.size() # स्क्रीन का साइज़ एक बार प्राप्त करें

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
            if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 1][1]:
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

            # ===== नया फीचर: लेजर पॉइंटर मोड =====
            # अगर सिर्फ तर्जनी उठी हुई है
            if total_fingers == 1 and fingers[1] == 1:
                cv2.putText(image, 'Pointer Mode', (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                # तर्जनी के सिरे (landmark 8) के कोऑर्डिनेट्स प्राप्त करें
                index_x, index_y = landmark_list[8][1], landmark_list[8][2]
                
                # स्क्रीन पर एक छोटा सा वृत्त बनाएं जहाँ पॉइंटर है
                cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), cv2.FILLED)

                # कैमरे के कोऑर्डिनेट्स को स्क्रीन के कोऑर्डिनेट्स में बदलें
                # np.interp(value, [input_range_start, input_range_end], [output_range_start, output_range_end])
                screen_x = np.interp(index_x, [w*0.2, w*0.8], [0, screen_w]) # थोड़ी पैडिंग ताकि कोनों में न जाए
                screen_y = np.interp(index_y, [h*0.2, h*0.8], [0, screen_h])
                
                # माउस को उस पोजीशन पर ले जाएं
                pyautogui.moveTo(screen_x, screen_y)

            # ===== स्वाइप जेस्चर (जब 5 उंगलियां खुली हों) =====
            elif total_fingers == 5:
                current_time = time.time()
                if current_time - last_gesture_time > gesture_cooldown:
                    wrist_landmark = my_hand.landmark[mp_hands.HandLandmark.WRIST]
                    if initial_wrist_x is None:
                        initial_wrist_x = wrist_landmark.x
                    else:
                        delta_x = wrist_landmark.x - initial_wrist_x
                        if delta_x > swipe_threshold:
                            pyautogui.press('right')
                            print("Right Swipe -> Next Slide")
                            last_gesture_time = current_time
                            initial_wrist_x = None
                        elif delta_x < -swipe_threshold:
                            pyautogui.press('left')
                            print("Left Swipe -> Previous Slide")
                            last_gesture_time = current_time
                            initial_wrist_x = None
                else:
                    cv2.putText(image, 'SWIPE COOLDOWN', (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 165, 0), 2)

    cv2.imshow("Gesture Control", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()