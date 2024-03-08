import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Load image
image_path = 'four/1.jpg'  # Fill in the path to your image
image = cv2.imread(image_path)
image = cv2.flip(image,1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect hands in the image
results = hands.process(image)
prev = curr = None
# Extract hand bounding box and hand nodes

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract coordinates of hand landmarks
        landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])

        # Get the bounding box coordinates of the detected hand
        nodes = np.array(landmarks * np.array(image.shape[1::-1]), dtype=np.int32)
        print([*nodes])
        # hand_rect = 
        x, y, width, height = cv2.boundingRect(nodes)
        h_offset = height//10
        w_offset = width//10
        print(x, y, width, height, h_offset, w_offset)
        blank_slate = np.zeros((height + (2 * h_offset), width + (2 * w_offset), 3), dtype = np.uint8)

        # Extract hand region
        hand_region = image[y:y + height, x:x + width]

        for landmark in landmarks:
            lx, ly = landmark
            
            cv2.circle(blank_slate, (int(lx * image.shape[1] - x + w_offset), int(ly * image.shape[0]) - y + h_offset), 3, (255, 255, 255), -1)

            if curr is None:
                curr = (int(lx * image.shape[1] - x + w_offset), int(ly * image.shape[0] - y) + h_offset)
            else:
                prev = curr  
                curr = (int(lx * image.shape[1] - x + w_offset), int(ly * image.shape[0] - y) + h_offset)

            if prev is not None:
                cv2.line(blank_slate, prev, curr, (255,255,255), 1)
        
        blank_slate = cv2.resize(blank_slate, (0,0), fx= 2, fy=2) # zoom
        blank_slate = cv2.cvtColor(blank_slate, cv2.COLOR_BGR2GRAY)
        blank_slate = cv2.threshold(blank_slate, 0.5, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("1", blank_slate)
        cv2.imwrite("new.jpg", blank_slate)
else:
    print("moye moye")
# Display bounding box, hand nodes, and connections
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
hands.close()