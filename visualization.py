import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Blueprint, render_template

visualization_blueprint = Blueprint('visualization', __name__, template_folder='templates')

# Load data from CSV file
try:
    data = pd.read_csv('predictions.csv')
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")

# Ensure the directory 'static/images' exists
images_dir = os.path.join('static', 'images')
os.makedirs(images_dir, exist_ok=True)
print(f"Images directory: {images_dir}")

@visualization_blueprint.route('/')
def visualization():
    # Generate plots
    bar_chart_age = generate_bar_chart_age()
    pie_chart_age = generate_pie_chart_age()
    bar_chart_gender = generate_bar_chart_gender()

    return render_template('visualization.html', 
                           bar_chart_age=bar_chart_age,
                           pie_chart_age=pie_chart_age,
                           bar_chart_gender=bar_chart_gender)

def generate_bar_chart_age():
    try:
        print("Generating age bar chart...")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Predicted_Age', data=data, palette='viridis')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        bar_chart_age = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        print("Age bar chart generated.")
        print(f"Age bar chart base64: {bar_chart_age[:50]}...")  # Print first 50 characters for verification
        return bar_chart_age
    except Exception as e:
        print(f"Error generating age bar chart: {e}")
        return ""

def generate_pie_chart_age():
    try:
        print("Generating age pie chart...")
        age_counts = data['Predicted_Age'].value_counts()

        # Create the pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.ylabel('Age Group Distribution')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        pie_chart_age = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        print("Age pie chart generated.")
        print(f"Age pie chart base64: {pie_chart_age[:50]}...")  # Print first 50 characters for verification
        return pie_chart_age
    except Exception as e:
        print(f"Error generating age pie chart: {e}")
        return ""

def generate_bar_chart_gender():
    try:
        print("Generating gender bar chart...")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Predicted_Gender', data=data, palette='viridis')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        bar_chart_gender = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        print("Gender bar chart generated.")
        print(f"Gender bar chart base64: {bar_chart_gender[:50]}...")  # Print first 50 characters for verification
        return bar_chart_gender
    except Exception as e:
        print(f"Error generating gender bar chart: {e}")
        return ""
