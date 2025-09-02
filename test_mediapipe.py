import cv2
import mediapipe as mp

print("Testing MediaPipe Initialization...")
print("Step 1: Initializing MediaPipe Hands. This might take a moment...")

try:
    # MediaPipe Hands ko initialize karne ki koshish karein
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    print("Step 2: MediaPipe Hands initialized SUCCESSFULLY. ✅")
    
    print("Step 3: Trying to open webcam...")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("Step 4: Webcam opened SUCCESSFULLY. ✅")
        print("\nTest Successful! Your MediaPipe and Webcam are working correctly.")
    else:
        print("Error: Could not open webcam. ❌")
        
    cap.release()

except Exception as e:
    print(f"\nAn error occurred during initialization: {e} ❌")