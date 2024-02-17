import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
from collections import deque
from statistics import mode

# Load the trained model and set to evaluation mode
model = torch.load('model.pth')
model.eval()

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define transformations and classes
transformations = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4725, 0.4614, 0.4750], std=[0.1818, 0.2227, 0.2390])
])
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space']  # Complete this list with your class names

# Initialize the camera
cap = cv2.VideoCapture(0)

# Stores the last 5 predictions for smoothing
last_predictions = deque(maxlen=5)


# Function to predict the class from an image
def predict_image(image, model, transformations):
    image = Image.fromarray(image)  # Convert frame to PIL Image
    image = transformations(image).float()
    image = image.unsqueeze(0)  # Add batch dimension
    output = model(image)
    probability, predicted = torch.max(torch.softmax(output, 1), 1)
    return classes[predicted.item()], probability.item()



# Main loop for the application
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform hand detection and get the bounding box
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the bounding box
            h, w, _ = frame.shape
            min_x, min_y = w, h
            max_x = max_y = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)
            # Expand the bounding box
            bbox_padding = 20
            min_x, min_y = max(min_x - bbox_padding, 0), max(min_y - bbox_padding, 0)
            max_x, max_y = min(w, max_x + bbox_padding), min(h, max_y + bbox_padding)
            # Crop the ROI from the frame
            hand_roi = frame[min_y:max_y, min_x:max_x]
            # Predict the hand sign using the cropped image
            hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
            prediction = predict_image(hand_roi_rgb, model, transformations)
            last_predictions.append(prediction)
            # Inside your main loop where predictions are made
            if len(last_predictions) == last_predictions.maxlen:
                smoothed_prediction, confidence = predict_image(hand_roi_rgb, model, transformations)
                last_predictions.append((smoothed_prediction, confidence))
                most_common = mode(last_predictions)
                cv2.putText(frame, f'{most_common[0]} ({most_common[1]:.2f})', (min_x, min_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, 'No hand detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
