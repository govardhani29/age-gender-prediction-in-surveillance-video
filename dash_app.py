import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

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
    html.Link(
        rel='stylesheet',
        href='/assets/styles.css'  # Link to the CSS file in the assets folder
    ),
    html.Div([
        html.H1("Age and Gender Prediction Dashboard", className="text-center mb-4"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.H4("Total Count", className="card-title"),
                    html.P(f"{total_count}", className="card-text")
                ], className="card-body"),
            ], className="card col-md-4"),
            
            html.Div([
                html.Div([
                    html.H4("Male Count", className="card-title"),
                    html.P(f"{male_count}", className="card-text")
                ], className="card-body"),
            ], className="card col-md-4"),
            
            html.Div([
                html.Div([
                    html.H4("Female Count", className="card-title"),
                    html.P(f"{female_count}", className="card-text")
                ], className="card-body"),
            ], className="card col-md-4")
        ], className="row mb-4"),
        
        html.Div([
            html.Div([
                html.H2("Gender Distribution", className="mb-3"),
                dcc.Graph(
                    id='gender-pie-chart',
                    figure=px.pie(names=['Male', 'Female'], values=[male_count, female_count], title='Gender Distribution')
                )
            ], className="col-lg-6"),
    
            html.Div([
                html.H2("Age Distribution", className="mb-3"),
                dcc.Graph(
                    id='age-bar-chart',
                    figure=px.bar(x=age_distribution.index, y=age_distribution.values, title='Age Distribution')
                )
            ], className="col-lg-6")
        ], className="row mb-4"),
    ], className="container")
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
