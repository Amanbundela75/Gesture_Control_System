import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

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

cap = cv2.VideoCapture(0)

gesture_cooldown = 1.5
last_gesture_time = 0
initial_wrist_x = None
swipe_threshold = 0.15
tip_ids = [4, 8, 12, 16, 20]
screen_w, screen_h = pyautogui.size()

drawing_mode_on = False  # 🔹 Drawing mode state

print("प्रोग्राम शुरू हो रहा है... Esc दबाकर बाहर निकलें।")


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
        continue

    h, w, c = image.shape
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
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
