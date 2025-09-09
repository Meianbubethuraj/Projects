import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import math

# Load the pre-trained model
model = load_model("C:/Users/19258/OneDrive/Desktop/language/model.h5")
labels = ["good", "hello", "I love You", "please"]  # Update labels to match your dataset

# Initialize the hand detector
detector = HandDetector(maxHands=1)

cap = cv2.VideoCapture(0)  # Open webcam (adjust the index if you have multiple cameras)

offset = 20
imgSize = 300
target_size = (150, 150)  # Target size expected by the model

# Open a text file to save predictions with accuracy
output_file = open("predictions.txt", "w")  # Open the file in write mode

# Set a confidence threshold
confidence_threshold = 0.8  # Increase confidence threshold

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    imgOutput = img.copy()
    
    # Find hands in the frame
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]  # Assume only one hand is detected
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Resize to the target size expected by the model
        imgWhite = cv2.resize(imgWhite, target_size)
        
        imgWhite = imgWhite / 255.0  # Normalize the image
        imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension

        # Predict gesture label and confidence
        prediction = model.predict(imgWhite)
        index = np.argmax(prediction)
        label = labels[index]
        confidence = prediction[0][index]

        # Check if the confidence is above the threshold
        if confidence >= confidence_threshold:
            display_label = f'{label} ({confidence:.2f})'
        else:
            display_label = 'Invalid Gesture'

        # Print the prediction in the console
        print(f'Prediction: {label}, Confidence: {confidence:.2f}')
        
        # Write prediction with accuracy to the text file
        output_file.write(f'Prediction: {display_label}\n')

        # Display the prediction on the OpenCV window
        cv2.putText(imgOutput, display_label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 255, 0), 2)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the text file
output_file.close()

cap.release()
cv2.destroyAllWindows()
