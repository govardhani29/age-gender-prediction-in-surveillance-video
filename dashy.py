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

    # Statistics Section
    html.Div([
        html.H2("Statistics", style={'text-align': 'center'}),
        html.Div(id="statistics-display", style={'margin': '20px'})
    ], style={'border': '1px solid #ddd', 'border-radius': '5px', 'margin': '20px', 'padding': '20px'}),

    # Plots Section
    html.Div([
        html.H2("Plots", style={'text-align': 'center'}),
        # Add a div for the bar chart
        dcc.Graph(id='bar-chart', style={'margin': '20px'}),
        # Add a div for the pie chart
        dcc.Graph(id='pie-chart', style={'margin': '20px'}),
        # Add a div for the histogram
        dcc.Graph(id='histogram', style={'margin': '20px'}),
        # Add a div for the box plot
        dcc.Graph(id='box-plot', style={'margin': '20px'})
    ], style={'border': '1px solid #ddd', 'border-radius': '5px', 'margin': '20px', 'padding': '20px'}),

    # Recent Data Section
    html.Div([
        html.Div([
            html.H2("Recent Data", style={'text-align': 'center', 'margin-bottom': '10px'}),
        ], style={'margin-bottom': '20px'}),
        html.Div(id="data-display", style={'margin': '20px'})
    ], style={'border': '1px solid #ddd', 'border-radius': '5px', 'margin': '20px', 'padding': '20px'}),
])

# Define callback to update data and graphs
@app.callback(
    [Output("data-display", "children"),
     Output("bar-chart", "figure"),
     Output("pie-chart", "figure"),
     Output("histogram", "figure"),
     Output("box-plot", "figure"),
     Output("statistics-display", "children")],
    [Input("update-button", "n_clicks")]
)
def update_data(n_clicks):
    # Read updated data from CSV
    updated_data = pd.read_csv('predictions.csv')

    # Select the most recent 10 rows of the data
    recent_data = updated_data.tail(10)

    # Generate bar chart
    bar_fig = px.bar(updated_data, x='Predicted_Age', color='Predicted_Gender', barmode='group',
                     title='Age Group Distribution by Gender')

    # Generate pie chart
    pie_fig = px.pie(updated_data, names='Predicted_Gender', title='Gender Distribution')

    # Generate histogram
    hist_fig = px.histogram(updated_data, x='Predicted_Age', color='Predicted_Gender', barmode='overlay',
                            title='Age Distribution')

    # Generate box plot
    box_fig = px.box(updated_data, x='Predicted_Gender', y='Predicted_Age', color='Predicted_Gender',
                     title='Age Distribution by Gender')

    # Calculate statistics
    total_count = len(updated_data)
    male_count = len(updated_data[updated_data['Predicted_Gender'] == 'man'])
    female_count = len(updated_data[updated_data['Predicted_Gender'] == 'woman'])
    age_group_counts = updated_data['Predicted_Age'].value_counts().to_dict()

    # Display updated data and statistics
    data_table = html.Div([
        html.Table([
            html.Thead(html.Tr([html.Th(col) for col in recent_data.columns])),
            html.Tbody([
                html.Tr([html.Td(recent_data.iloc[i][col]) for col in recent_data.columns])
                for i in range(len(recent_data))  # Display recent 10 rows
            ])
        ], style={'width': '100%', 'border': '1px solid black', 'border-collapse': 'collapse'})
    ])

    statistics = html.Div([
        html.P(f"Total People: {total_count}"),
        html.P(f"Number of Males: {male_count}"),
        html.P(f"Number of Females: {female_count}"),
        html.P("Age Group Distribution:"),
        html.Ul([html.Li(f"{age_group}: {count}") for age_group, count in age_group_counts.items()])
    ])

    return data_table, bar_fig, pie_fig, hist_fig, box_fig, statistics

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
