import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import math
import time

# ==============================
# Global Setup
# ==============================
# MediaPipe Hands Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Frame and Screen Dimensions
screen_w, screen_h = pyautogui.size()
frame_w, frame_h = 640, 480

# Gesture Control Variables
tip_ids = [4, 8, 12, 16, 20]
gesture_cooldown = 1.0  # Cooldown between gestures
last_gesture_time = 0
initial_wrist_x = None
swipe_threshold = 0.15
drawing_mode_on = False

# ==============================
# Helper Functions (Defined Before the Loop)
# ==============================

def count_fingers(landmarks):
    """Returns a list of finger states [1 for up, 0 for down] and the total count of fingers up."""
    fingers = []
    # Thumb (checks horizontal position relative to its base)
    if landmarks[tip_ids[0]][1] > landmarks[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other 4 fingers (checks vertical position relative to their base)
    for id in range(1, 5):
        if landmarks[tip_ids[id]][2] < landmarks[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers, fingers.count(1)

def pointer_mode(landmarks, image):
    """Moves the mouse cursor based on the index finger's tip."""
    cv2.putText(image, 'Pointer Mode', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    index_x, index_y = landmarks[8][1], landmarks[8][2]
    cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), cv2.FILLED)
    # Interpolate coordinates for the screen
    screen_x = np.interp(index_x, [100, frame_w - 100], [0, screen_w])
    screen_y = np.interp(index_y, [100, frame_h - 100], [0, screen_h])
    pyautogui.moveTo(screen_x, screen_y)

def swipe_gesture(my_hand):
    """Performs a left or right keypress based on horizontal wrist movement."""
    global initial_wrist_x, last_gesture_time
    wrist_landmark = my_hand.landmark[mp_hands.HandLandmark.WRIST]
    if initial_wrist_x is None:
        initial_wrist_x = wrist_landmark.x
    else:
        delta_x = wrist_landmark.x - initial_wrist_x
        if delta_x > swipe_threshold:
            pyautogui.press('right')
            print("Right Swipe")
            last_gesture_time = time.time()
            initial_wrist_x = None
        elif delta_x < -swipe_threshold:
            pyautogui.press('left')
            print("Left Swipe")
            last_gesture_time = time.time()
            initial_wrist_x = None

def volume_control(landmarks, image):
    """Adjusts volume based on the distance between wrist and a finger knuckle."""
    global last_gesture_time
    cv2.putText(image, 'Volume Mode', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    wrist = landmarks[0][1], landmarks[0][2]
    mcp_middle = landmarks[9][1], landmarks[9][2]
    hand_size = math.hypot(mcp_middle[0] - wrist[0], mcp_middle[1] - wrist[1])
    # Map hand size to volume
    vol = np.interp(hand_size, [50, 200], [0, 100])
    print(f"Hand Size: {hand_size}, Volume: {vol}") # For debugging
    # Note: Volume control is highly OS-dependent. PyAutoGUI presses keys.
    if hand_size > 150:
        pyautogui.press('volumedown')
    elif hand_size < 80:
        pyautogui.press('volumeup')
    last_gesture_time = time.time()

def take_screenshot(image):
    """Takes and saves a screenshot."""
    global last_gesture_time
    cv2.putText(image, 'Screenshot!', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pyautogui.screenshot(f"screenshot_{timestamp}.png")
    print(f"Screenshot saved.")
    last_gesture_time = time.time()

def drawing_mode(landmarks, image):
    """Controls mouse down/up for drawing."""
    global drawing_mode_on
    cv2.putText(image, 'Drawing Mode', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    index_tip = landmarks[8][1], landmarks[8][2]
    middle_tip = landmarks[12][1], landmarks[12][2]
    # Move mouse to index finger tip
    screen_x = np.interp(index_tip[0], [100, frame_w - 100], [0, screen_w])
    screen_y = np.interp(index_tip[1], [100, frame_h - 100], [0, screen_h])
    pyautogui.moveTo(screen_x, screen_y)
    # Check distance between index and middle finger
    distance = math.hypot(middle_tip[0] - index_tip[0], middle_tip[1] - index_tip[1])
    if distance < 40:  # If fingers are close
        if not drawing_mode_on:
            pyautogui.mouseDown()
            drawing_mode_on = True
            print("Mouse Down")
        cv2.circle(image, index_tip, 10, (0, 255, 0), cv2.FILLED) # Green circle for drawing
    else: # If fingers are apart
        if drawing_mode_on:
            pyautogui.mouseUp()
            drawing_mode_on = False
            print("Mouse Up")
        cv2.circle(image, index_tip, 10, (0, 0, 255), cv2.FILLED) # Red circle for moving

# ==============================
# Main Loop
# ==============================
cap = cv2.VideoCapture(0)
cap.set(3, frame_w)
cap.set(4, frame_h)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB before processing.
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        my_hand = results.multi_hand_landmarks[0]
        
        # Get landmark coordinates
        landmark_list = []
        for id, lm in enumerate(my_hand.landmark):
            cx, cy = int(lm.x * frame_w), int(lm.y * frame_h)
            landmark_list.append([id, cx, cy])
        
        if landmark_list:
            fingers, total_fingers = count_fingers(landmark_list)
            
            # Draw landmarks on the image
            mp_draw.draw_landmarks(image, my_hand, mp_hands.HAND_CONNECTIONS)
            
            # Check for cooldown
            current_time = time.time()
            if (current_time - last_gesture_time) > gesture_cooldown:
                # Decision logic based on finger count
                if total_fingers == 1 and fingers[1] == 1:
                    pointer_mode(landmark_list, image)
                elif total_fingers == 5:
                    swipe_gesture(my_hand)
                elif total_fingers == 0:
                    volume_control(landmark_list, image)
                elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                    drawing_mode(landmark_list, image)
                elif total_fingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    take_screenshot(image) # Screenshot with 3 fingers
            else:
                # Reset initial wrist position if not in swipe gesture
                initial_wrist_x = None

    # Display the resulting frame
    cv2.imshow("Gesture Control", image)

    # Exit on 'Esc' key
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()