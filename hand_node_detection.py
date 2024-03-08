import cv2
import mediapipe as mp

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

# print(list(results.multi_hand_landmarks))
# Extract hand nodes
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Iterate through hand landmarks and convert them to node versions
        for landmark in hand_landmarks.landmark:
            # Convert landmarks to node versions
            # For simplicity, we'll draw circles at the positions of the landmarks
            image_height, image_width, _ = image.shape
            cx, cy = int(landmark.x * image_width), int(landmark.y * image_height)
            # Draw node (circle)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # You can adjust the radius and color as needed

# Display the result
cv2.imshow('Hand Nodes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
hands.close()