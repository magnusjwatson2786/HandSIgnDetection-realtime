import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Load image
image_path = 'palms/1.jpg'  # Fill in the path to your image
image = cv2.imread(image_path)

# Convert image to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect hands in the image
results = hands.process(image)

print(len(results.multi_hand_landmarks))
hand_regions = []

# Extract hand bounding box and hand nodes
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract coordinates of hand landmarks
        landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])

        # Draw hand nodes
        for landmark in landmarks:
            x, y = landmark
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 2, (255, 0, 0), -1)

        # Connect hand landmarks with lines
        connections = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15],
                       [15, 16], [17, 18], [18, 19], [19, 20]]
        for connection in connections:
            x_start, y_start = landmarks[connection[0]]
            x_end, y_end = landmarks[connection[1]]
            cv2.line(image, (int(x_start * image.shape[1]), int(y_start * image.shape[0])),
                     (int(x_end * image.shape[1]), int(y_end * image.shape[0])), (0, 255, 0), 1)

        # Get the bounding box coordinates of the detected hand
        hand_rect = cv2.boundingRect(np.array(landmarks * np.array(image.shape[1::-1]), dtype=np.int32))

        # Extract hand region
        hand_region = image[hand_rect[1]:hand_rect[1] + hand_rect[3], hand_rect[0]:hand_rect[0] + hand_rect[2]]
        hand_region = cv2.resize(hand_region, (0,0), fx= 2, fy=2)
        hand_regions.append(hand_region)

# Display bounding box, hand nodes, and connections
print(len(hand_regions))
for i in range(len(hand_regions)):
    cv2.imshow(str(i+1), hand_regions[i])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
hands.close()