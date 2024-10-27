import cv2
from google.colab.patches import cv2_imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# Constants for resizing images
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Resize and show function for Colab environment
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2_imshow(img)

# STEP 1: Load the Gesture Recognizer model
base_options = python.BaseOptions(model_asset_path='models\gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Prepare the images and recognition results
images = []
results = []
for image_file_name in IMAGE_FILENAMES:
    # Load the image using mediapipe
    image = mp.Image.create_from_file('Vu2Nqwb.jpeg')
    recognition_result = recognizer.recognize(image)

    # Convert mediapipe image to OpenCV format
    cv2_image = cv2.imread(image_file_name)
    images.append(cv2_image)
    
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))

# Display function
def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    for idx, (image, (top_gesture, hand_landmarks)) in enumerate(zip(images, results)):
        # Draw the gesture text on the image
        cv2.putText(image, f"Gesture: {top_gesture.category_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw the hand landmarks
        for point in hand_landmarks[0]:  # Assuming one hand
            x, y = int(point.x * image.shape[1]), int(point.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        
        # Resize and display the image in Colab
        resize_and_show(image)

# Call the function to display images
display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
