import cv2
import mediapipe as mp
import numpy as np
# from google.colab.patches import cv2_imshow

# Load MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=5)

# Load image
image_path = 'palms/55.jpg'  # Fill in the path to your image
image = cv2.imread(image_path)

# Convert image to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect hands in the image
results = hands.process(image)
hand_regions = []
# Extract hand bounding box
print(len(results.multi_hand_landmarks))
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract coordinates of hand landmarks
        landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
        # print(landmarks)
        # Get the bounding box coordinates of the detected hand
        # print(image.shape[1::-1])
        hand_rect = cv2.boundingRect(np.array(landmarks * np.array(image.shape[1::-1]), dtype=np.int32))

        # Draw bounding box
        cv2.rectangle(image, (hand_rect[0], hand_rect[1]), (hand_rect[0] + hand_rect[2], hand_rect[1] + hand_rect[3]),
                      (0, 255, 0), 2)

        # Extract hand region
        hand_region = image[hand_rect[1]:hand_rect[1] + hand_rect[3], hand_rect[0]:hand_rect[0] + hand_rect[2]]
        hand_region = cv2.resize(hand_region, (0,0), fx= 2, fy=2)
        hand_regions.append(hand_region)
# Display bounding box and extracted hand region
# cv2.imshow("Hand Region", hand_region)  
# cv2.imshow("Hand Region", image)  
print(len(hand_regions))
for i in range(len(hand_regions)):
    cv2.imshow(str(i+1), hand_regions[i])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
hands.close()