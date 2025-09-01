import cv2
import os

# Dataset save karne ke liye main folder
DATA_DIR = './dataset'

# Agar dataset folder nahi hai, toh bana do
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Aap kitne gestures (classes) chahte hain aur har gesture ki kitni images?
num_of_classes = 3
num_of_samples = 200 # Har gesture ke liye 200 images

# Camera shuru karo
cap = cv2.VideoCapture(0)

# Har gesture ke liye folder banao aur images collect karo
for i in range(num_of_classes):
    
    # Gesture ke liye folder banao
    class_name = input(f'Enter name for Gesture Class {i+1}: ') # Jaise 'Peace', 'Fist'
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting images for {class_name}...')
    
    # User ko taiyaar hone ke liye bolo
    ready_text = f"Ready? Show '{class_name}'. Press 'S' to start."
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, ready_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break

    # Images save karna shuru karo
    counter = 0
    while counter < num_of_samples:
        ret, frame = cap.read()
        
        # Image save karo
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)

        # Screen par counter dikhao
        cv2.putText(frame, f'Collecting: {counter}/{num_of_samples}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(25) # Thoda delay
        
        counter += 1

cap.release()
cv2.destroyAllWindows()