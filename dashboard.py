import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import os
import csv

# Load gender detection model
gender_model = load_model('gender_detection.h5')

# Load age detection model
age_model = load_model('age_detection.h5')

# Define gender and age categories
gender_classes = ['man', 'woman']
age_categories = ['0-12', '13-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']

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

# Function to perform gender and age detection
def detect_gender_age():
    # Open webcam
    webcam = cv2.VideoCapture(0)

    # Read frame from webcam 
    status, frame = webcam.read()

    if status:
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

        # Release resources
        webcam.release()

# Load data
data = pd.read_csv('predictions.csv')

# Calculate statistics
total_count = len(data)
male_count = (data['Predicted_Gender'] == 'man').sum()
female_count = (data['Predicted_Gender'] == 'woman').sum()
age_distribution = data['Predicted_Age'].value_counts()

# Define app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Age and Gender Prediction Dashboard"),
    
    html.Button("Detect Gender and Age", id="detect-button", n_clicks=0),
    
    html.Div([
        html.Div([
            html.H4("Total Count"),
            html.P(f"{total_count}")
        ], className="card-body"),

        html.Div([
            html.H4("Male Count"),
            html.P(f"{male_count}")
        ], className="card-body"),

        html.Div([
            html.H4("Female Count"),
            html.P(f"{female_count}")
        ], className="card-body")
    ], className="row mb-4"),

    html.Div([
        html.Div([
            html.H2("Gender Distribution"),
            dcc.Graph(id='gender-pie-chart')
        ], className="col-lg-6"),

        html.Div([
            html.H2("Age Distribution"),
            dcc.Graph(id='age-bar-chart')
        ], className="col-lg-6")
    ], className="row")
])

# Callback to detect gender and age on button click
@app.callback(
    Output('gender-pie-chart', 'figure'),
    Output('age-bar-chart', 'figure'),
    Input('detect-button', 'n_clicks')
)
def update_charts(n_clicks):
    print("Button clicked:", n_clicks)
    if n_clicks > 0:
        detect_gender_age()
        print("Detection completed")
        # Reload data
        data = pd.read_csv('predictions.csv')
        print("Data loaded")
        # Update statistics
        total_count = len(data)
        male_count = (data['Predicted_Gender'] == 'man').sum()
        female_count = (data['Predicted_Gender'] == 'woman').sum()
        age_distribution = data['Predicted_Age'].value_counts()
        print("Statistics updated")
        # Update charts
        gender_fig = px.pie(names=['Male', 'Female'], values=[male_count, female_count], title='Gender Distribution')
        age_fig = px.bar(x=age_distribution.index, y=age_distribution.values, title='Age Distribution')
        print("Charts updated")
        return gender_fig, age_fig
    else:
        return {}, {}

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
