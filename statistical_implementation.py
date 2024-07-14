from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import os
import csv

# Function to append a new prediction to the CSV file
def append_prediction_to_csv(age, gender, filename='predictions.csv'):
    # Check if the file exists
    if not os.path.isfile(filename):
        # If file does not exist, create it with headers
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Predicted_Age', 'Predicted_Gender'])

    # Append the new prediction
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([age, gender])

# Load gender detection model
gender_model = load_model('gender_detection.h5')

# Load age detection model
age_model = load_model('age_detection.h5')

# Define gender and age categories
gender_classes = ['man', 'woman']
age_categories = ['0-12', '13-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']

# Open webcam
webcam = cv2.VideoCapture(0)

# Define the folder path where you want to save the videos
output_folder = r'C:\Users\91951\Downloads\agegender'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_path = os.path.join(output_folder, 'output.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

# Loop through frames
while webcam.isOpened():

    # Read frame from webcam 
    status, frame = webcam.read()

    if not status:
        break

    # Resize frame to match the VideoWriter resolution
    frame = cv2.resize(frame, (640, 480))

    # Apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):

        # Crop the detected face region
        face_crop = np.copy(frame[f[1]:f[3], f[0]:f[2]])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender and age detection models
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        gender_conf = gender_model.predict(face_crop)[0]
        gender_idx = np.argmax(gender_conf)
        gender_label = gender_classes[gender_idx]

        # Apply age detection on face
        age_conf = age_model.predict(face_crop)[0]
        age_idx = np.argmax(age_conf)
        age_label = age_categories[age_idx]

        # Append prediction to CSV
        append_prediction_to_csv(age_label, gender_label)

        # Draw rectangle over face
        cv2.rectangle(frame, (f[0], f[1]), (f[2], f[3]), (0, 255, 0), 2)

        # Write labels and confidence above face rectangle
        gender_text = "{}".format(gender_label)
        age_text = "{}".format(age_label)
        cv2.putText(frame, gender_text, (f[0], f[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, age_text, (f[0], f[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write the frame into the file
    out.write(frame)

    # Display output
    cv2.imshow("Gender and Age Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
out.release()
cv2.destroyAllWindows()
