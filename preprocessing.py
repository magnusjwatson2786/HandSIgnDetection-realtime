import cv2, os
import mediapipe as mp
import numpy as np

# Load MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def preprocess(image_path, fails):
    image_name = image_path.split('/')[-1] # filename
    image_class = image_path.split('/')[-2] # folder name
    if not os.path.exists('preprocessed_data/'+ image_class):
        os.makedirs('preprocessed_data/'+ image_class)
    dest_path = 'preprocessed_data/'+ image_class + '/' + image_name
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
            x, y, width, height = cv2.boundingRect(nodes)
            h_offset = height//10
            w_offset = width//10
            # print(x, y, width, height, h_offset, w_offset)
            blank_slate = np.zeros((height + (2 * h_offset), width + (2 * w_offset)), dtype = np.uint8) # 2D array instead of 3D

            for landmark in landmarks:
                lx, ly = landmark
                cv2.circle(blank_slate, (int(lx * image.shape[1] - x + w_offset), int(ly * image.shape[0]) - y + h_offset), 3, 255, -1) # 1D color
                if curr is None:
                    curr = (int(lx * image.shape[1] - x + w_offset), int(ly * image.shape[0] - y) + h_offset)
                else:
                    prev = curr  
                    curr = (int(lx * image.shape[1] - x + w_offset), int(ly * image.shape[0] - y) + h_offset)
                if prev is not None:
                    cv2.line(blank_slate, prev, curr, 255, 1) # 1D color
            # blank_slate = cv2.resize(blank_slate, (0,0), fx= 2, fy=2) # zoom disabled
            # blank_slate = cv2.threshold(blank_slate, 1, 255, cv2.THRESH_BINARY)[1] # not really needed
            # cv2.imshow("1", blank_slate)
            cv2.imwrite(dest_path, blank_slate)
            # print(dest_path)
    else:
        fails.append(image_path)

def printProgressBar (iteration, total, prefix = '', suffix = '', length = 100, fill = '█', printEnd = "\r"):
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {iteration} {suffix}', end = printEnd)
    if iteration == total: 
        print()

done = os.listdir('preprocessed_data/')
total = os.listdir('hagrid_dataset_512/')
not_done = [ i for i in total if i not in done]
print(not_done)
parent_path = 'hagrid_dataset_512/'
for i in not_done:
    path = parent_path + i +'/'
    files = os.listdir(path)
    count = 0
    total = len(files)
    failed = []
    printProgressBar(count, total, i, 'image(s) sucessfully converted', 60)
    for j in files:
        preprocess(path+j, failed)
        count += 1
        printProgressBar(count, total, i, 'image(s) sucessfully converted', 60)
    print(f'{len(failed)} images couldnt be converted')
    with open('failed/'+i+'-failed.txt', 'w') as f:
        for item in failed:
            f.write(item + '\n')


cv2.waitKey(0)
cv2.destroyAllWindows()
# Release resources
hands.close()