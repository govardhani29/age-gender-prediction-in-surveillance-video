import dash
from dash import html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Age and Gender Prediction Dashboard", style={'text-align': 'center'}),

    # Add a button to manually update data
    html.Button("Update Data", id="update-button", n_clicks=0, style={'margin': '20px'}),

    # Display the latest data
    html.Div(id="data-display"),

    # Add a div for the bar chart
    dcc.Graph(id='bar-chart'),

    # Add a div for the pie chart
    dcc.Graph(id='pie-chart'),

    # Add a div for the statistics
    html.Div(id="statistics-display", style={'margin': '20px'}),
])

# Define callback to update data and graphs
@app.callback(
    [Output("data-display", "children"),
     Output("bar-chart", "figure"),
     Output("pie-chart", "figure"),
     Output("statistics-display", "children")],
    [Input("update-button", "n_clicks")]
)
def update_data(n_clicks):
    # Read updated data from CSV
    updated_data = pd.read_csv('predictions.csv')

    # Generate bar chart
    bar_fig = px.bar(updated_data, x='Predicted_Age', color='Predicted_Gender', barmode='group',
                     title='Age Group Distribution by Gender')

    # Generate pie chart
    pie_fig = px.pie(updated_data, names='Predicted_Gender', title='Gender Distribution')

    # Calculate statistics
    total_count = len(updated_data)
    male_count = len(updated_data[updated_data['Predicted_Gender'] == 'man'])
    female_count = len(updated_data[updated_data['Predicted_Gender'] == 'woman'])
    age_group_counts = updated_data['Predicted_Age'].value_counts().to_dict()

    # Display updated data and statistics
    data_table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in updated_data.columns])),
        html.Tbody([
            html.Tr([html.Td(updated_data.iloc[i][col]) for col in updated_data.columns])
            for i in range(min(len(updated_data), 10))  # Display first 10 rows
        ])
    ])

    statistics = html.Div([
        html.H3("Statistics"),
        html.P(f"Total People: {total_count}"),
        html.P(f"Number of Males: {male_count}"),
        html.P(f"Number of Females: {female_count}"),
        html.P("Age Group Distribution:"),
        html.Ul([html.Li(f"{age_group}: {count}") for age_group, count in age_group_counts.items()])
    ])

    return data_table, bar_fig, pie_fig, statistics

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
