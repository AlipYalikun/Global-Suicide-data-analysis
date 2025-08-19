import os
import base64
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
from scipy.stats import probplot, shapiro
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


#----------------------------------------------------------#
# web link https://dashapp-473886336048.us-east1.run.app/  #
#----------------------------------------------------------#

warnings.filterwarnings('ignore')
file_path = 'suicide_rate.csv'
data = pd.read_csv(file_path)
data['SuicideRatePer100k'] = data['SuicideCount'] / data['Population'] * 100000
data = data.dropna()

app = dash.Dash('__name__',external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Data Analysis Dashboard"

# Tabs Layout
app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Data Cleaning', children=[
            html.H3('Data Cleaning Methods'),
            html.Div(id='cleaning-output', style={'margin-bottom': '20px'}),
            html.Button('Clean Data', id='clean-button', n_clicks=0),
            dcc.Graph(id='random-plot')
        ]),
        dcc.Tab(label='Outlier Detection and Removal', children=[
            html.H3('Outlier Detection and Removal Using IQR'),
            html.Div(id='outlier-output', style={'margin-bottom': '20px'}),
            html.Button('Remove Outliers', id='remove-outliers-button', n_clicks=0),
            dcc.Graph(id='before-box-plot'),
            dcc.Graph(id='after-box-plot')
        ]),
        dcc.Tab(label='Dimensionality Reduction', children=[
            html.H3('Dimensionality Reduction with PCA'),
            html.Div(id='pca-output', style={'margin-bottom': '20px'}),
            html.Button('Run PCA', id='run-pca-button', n_clicks=0),
            dcc.Graph(id='pca-plot')
        ]),
        dcc.Tab(label='Normality Tests', children=[
            html.H3('Normality Test and Transformation'),
            html.Div(id='normality-output', style={'margin-bottom': '20px'}),
            html.Button('Check Normality', id='check-normality-button', n_clicks=0),
            dcc.Graph(id='before-normality-plot'),
            dcc.Graph(id='after-normality-plot')
        ]),
        dcc.Tab(label='Data Transformation', children=[
            html.H3('Apply Data Transformation'),
            html.Label('Select Transformation Type:'),
            dcc.Dropdown(
                id='transformation-type-dropdown',
                options=[
                     {'label': 'Log Transformation', 'value': 'log'},
                    {'label': 'Square Root Transformation', 'value': 'sqrt'},
                    {'label': 'Min-Max Scaling', 'value': 'minmax'}
                ],
                placeholder="Select a transformation type"
            ),
            html.Div(id='transformation-output', style={'margin-bottom': '20px'}),
            html.Button('Apply Transformation', id='apply-transformation-button', n_clicks=0),
            dcc.Graph(id='before-transformation-plot'),
            dcc.Graph(id='after-transformation-plot')
        ]),
        dcc.Tab(label='Statistics', children=[
            html.H3('Dataset Statistics'),
            html.Label('Select Feature for Statistics:'),
            dcc.Dropdown(
                id='statistics-dropdown',
                options=[{'label': col, 'value': col} for col in
                         data.select_dtypes(include=['float64', 'int64']).columns],
                placeholder="Select a numeric column"
            ),
            html.Div(id='statistics-output', style={'margin-top': '20px'}),
            dcc.Graph(id='statistics-plot')
        ]),
        dcc.Tab(label='Dynamic Plotting', children=[
            html.Div([
                html.H3('Dynamic Plotting for All Features'),
                html.Label('Select Features:'),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in data.columns],
                    multi=True,
                    placeholder="Select feature(s)"
                ),
                html.Label('Group By (optional):'),
                dcc.Dropdown(
                    id='group-by-dropdown',
                    options=[{'label': col, 'value': col} for col in data.select_dtypes(include='object').columns],
                    multi=False,
                    placeholder="Select a grouping feature (categorical)"
                ),
                html.Label('Select Plot Type:'),
                dcc.Dropdown(
                    id='plot-type-dropdown',
                    options=[
                        {'label': 'Line Plot', 'value': 'line'},
                        {'label': 'Bar Plot (Grouped)', 'value': 'bar_grouped'},
                        {'label': 'Bar Plot (Stacked)', 'value': 'bar_stacked'},
                        {'label': 'Count Plot', 'value': 'count'},
                        {'label': 'Pie Chart', 'value': 'pie'},
                        {'label': 'Dist Plot', 'value': 'dist'},
                        {'label': 'Pair Plot', 'value': 'pair'},
                        {'label': 'Heatmap with CBar', 'value': 'heatmap'},
                        {'label': 'Histogram with KDE', 'value': 'hist_kde'},
                        {'label': 'QQ-Plot', 'value': 'qq'},
                        {'label': 'KDE Plot', 'value': 'kde'},
                        {'label': 'Reg Plot', 'value': 'reg'},
                        {'label': 'Box/Boxen Plot', 'value': 'box'},
                        {'label': 'Area Plot', 'value': 'area'},
                        {'label': 'Violin Plot', 'value': 'violin'},
                        {'label': 'Joint Plot', 'value': 'joint'},
                        {'label': 'Rug Plot', 'value': 'rug'},
                        {'label': '3D Plot', 'value': '3d'},
                        {'label': 'Contour Plot', 'value': 'contour'},
                        {'label': 'Cluster Map', 'value': 'cluster'},
                        {'label': 'Hexbin Plot', 'value': 'hexbin'},
                        {'label': 'Strip Plot', 'value': 'strip'},
                        {'label': 'Swarm Plot', 'value': 'swarm'}
                    ],
                    value='line',
                    placeholder="Select Plot Type"
                ),
                dcc.Graph(id='dynamic-plot')
            ])
        ])
    ])
])

#call back for cleaning
@app.callback(
    [Output('cleaning-output', 'children'),
     Output('random-plot', 'figure')],
    [Input('clean-button', 'n_clicks')]
)
def clean_data(n_clicks):
    global data
    duplicates_count = data.duplicated().sum()
    nans_count = data.isnull().sum().sum()

    if n_clicks is None or n_clicks == 0:
        return [
            f"Dataset contains {duplicates_count} duplicate rows and {nans_count} missing values.",
            go.Figure()
        ]

    data = data.drop_duplicates()
    data = data.dropna()

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_columns.any():
        return [
            "All duplicates and missing values have been removed. However, no numeric columns are available for plotting.",
            go.Figure()
        ]

    random_feature = random.choice(numeric_columns)
    fig = px.histogram(data, x=random_feature, title=f"Distribution of {random_feature}")

    return [
        f"All duplicates and missing values have been removed. Dataset is now clean.",
        fig
    ]

# call backs for outlier
@app.callback(
    [Output('outlier-output', 'children'),
     Output('before-box-plot', 'figure'),
     Output('after-box-plot', 'figure')],
    [Input('remove-outliers-button', 'n_clicks')]
)
def remove_outliers(n_clicks):
    global data
    suicide_counts = data['SuicideCount']

    Q1 = suicide_counts.quantile(0.25)
    Q3 = suicide_counts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    if n_clicks is None or n_clicks == 0:
        fig_before = px.box(data, y='SuicideCount', title="Box Plot Before Removing Outliers")
        return [
            f"Initial outliers detected outside the range [{lower_bound:.2f}, {upper_bound:.2f}].",
            fig_before,
            go.Figure()
        ]

    data_cleaned = data[(suicide_counts >= lower_bound) & (suicide_counts <= upper_bound)]

    fig_before = px.box(data, y='SuicideCount', title="Box Plot Before Removing Outliers")
    fig_after = px.box(data_cleaned, y='SuicideCount', title="Box Plot After Removing Outliers")

    data = data_cleaned

    return [
        f"Outliers have been removed. Data points outside the range [{lower_bound:.2f}, {upper_bound:.2f}] were removed.",
        fig_before,
        fig_after
    ]

# call back for pca
@app.callback(
    [Output('pca-output', 'children'),
     Output('pca-plot', 'figure')],
    [Input('run-pca-button', 'n_clicks')]
)
def perform_pca(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return "Click 'Run PCA' to start dimensionality reduction.", go.Figure()

    numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_data)

    pca = PCA()
    pca.fit(standardized_data)

    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    num_features_95 = (cumulative_variance >= 0.95).argmax() + 1

    fig = px.line(
        x=range(1, len(cumulative_variance) + 1),
        y=cumulative_variance,
        title="Cumulative Variance Explained by Principal Components",
        labels={"x": "Number of Components", "y": "Cumulative Variance"},
    )
    fig.add_shape(
        type="line",
        x0=num_features_95,
        x1=num_features_95,
        y0=0,
        y1=1,
        line=dict(color="red", dash="dash"),
        xref="x",
        yref="y",
    )
    fig.add_annotation(
        x=num_features_95,
        y=0.95,
        text=f"95% Variance ({num_features_95} Components)",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-30,
    )

    return (
        f"{num_features_95} components are required to retain 95% of the variance.",
        fig,
    )

# call back for sharpiro normality test
@app.callback(
    [Output('normality-output', 'children'),
     Output('before-normality-plot', 'figure'),
     Output('after-normality-plot', 'figure')],
    [Input('check-normality-button', 'n_clicks')]
)

def check_and_transform_normality(n_clicks):
    global data
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) == 0:
        return (
            "No numeric columns available for normality testing.",
            go.Figure(),
            go.Figure()
        )
    selected_column = numeric_columns[4]
    if n_clicks is None or n_clicks == 0:
        fig_before = px.histogram(data, x=selected_column, title=f"Distribution of {selected_column} (Before)")
        return (
            f"Normality test for {selected_column} has not been performed yet.",
            fig_before,
            go.Figure()
        )
    stat, p_value = shapiro(data[selected_column].dropna())
    is_normal = p_value > 0.05

    if not is_normal:
        data[f"{selected_column}_transformed"] = np.log1p(data[selected_column])
        fig_after = px.histogram(data, x=f"{selected_column}_transformed",
                                 title=f"Transformed Distribution of {selected_column} (After)")
        message = (
            f"The column {selected_column} is not normally distributed (p-value = {p_value:.4f}). "
            "A log transformation has been applied."
        )
    else:
        fig_after = go.Figure()
        message = (
            f"The column {selected_column} is already normally distributed (p-value = {p_value:.4f}). "
            "No transformation was applied."
        )

    fig_before = px.histogram(data, x=selected_column, title=f"Distribution of {selected_column} (Before)")

    return message, fig_before, fig_after


# callback for minmax transformation
@app.callback(
    [Output('transformation-output', 'children'),
     Output('before-transformation-plot', 'figure'),
     Output('after-transformation-plot', 'figure')],
    [Input('transformation-type-dropdown', 'value'),
     Input('apply-transformation-button', 'n_clicks')]
)
def apply_transformation(transformation_type, n_clicks):
    global data
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) == 0:
        return (
            "No numeric columns available for transformation.",
            go.Figure(),
            go.Figure()
        )
    selected_column = numeric_columns[4]
    if n_clicks is None or n_clicks == 0:
        fig_before = px.histogram(data, x=selected_column, title=f"Distribution of {selected_column} (Before)")
        return (
            f"No transformation applied yet. Select a type of transformation and click 'Apply Transformation'.",
            fig_before,
            go.Figure()
        )
    if transformation_type == 'log':
        data[f"{selected_column}_transformed"] = np.log1p(data[selected_column])
        transformation_msg = "Log transformation applied."
    elif transformation_type == 'sqrt':
        data[f"{selected_column}_transformed"] = np.sqrt(data[selected_column])
        transformation_msg = "Square root transformation applied."
    elif transformation_type == 'minmax':
        scaler = MinMaxScaler()
        data[f"{selected_column}_transformed"] = scaler.fit_transform(data[[selected_column]])
        transformation_msg = "Min-max scaling applied."
    else:
        return (
            "Please select a valid transformation type.",
            go.Figure(),
            go.Figure()
        )
    fig_before = px.histogram(data, x=selected_column, title=f"Distribution of {selected_column} (Before)")
    fig_after = px.histogram(data, x=f"{selected_column}_transformed",
                             title=f"Distribution of {selected_column} (After {transformation_type.capitalize()})")
    return (
        f"{transformation_msg} Transformation applied to {selected_column}.",
        fig_before,
        fig_after
    )



# call back for stats
@app.callback(
    [Output('statistics-output', 'children'),
     Output('statistics-plot', 'figure')],
    Input('statistics-dropdown', 'value')
)
def display_statistics(selected_feature):
    if not selected_feature:
        return "Please select a feature to display statistics.", go.Figure()
    stats = data[selected_feature].describe()
    stats_output = html.Div([
        html.P(f"Mean: {stats['mean']:.2f}"),
        html.P(f"Median: {data[selected_feature].median():.2f}"),
        html.P(f"Standard Deviation: {stats['std']:.2f}"),
        html.P(f"Minimum: {stats['min']:.2f}"),
        html.P(f"Maximum: {stats['max']:.2f}"),
        html.P(f"25th Percentile: {stats['25%']:.2f}"),
        html.P(f"75th Percentile: {stats['75%']:.2f}")
    ])
    fig = px.box(data, y=selected_feature, title=f"Box Plot for {selected_feature}")
    return stats_output, fig


# Callbacks for dynamic plotting
@app.callback(
    Output('dynamic-plot', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('group-by-dropdown', 'value'),
     Input('plot-type-dropdown', 'value')]
)
def update_plot(selected_features, group_by, plot_type):
    if not selected_features:
        return go.Figure()

    # Handle different plot types
    if plot_type == 'line':
        if group_by:
            fig = px.line(data, x=selected_features[0], y=selected_features[1], color=group_by)
        else:
            fig = px.line(data, x=selected_features[0], y=selected_features[1])
    elif plot_type in ['bar_grouped', 'bar_stacked']:
        barmode = 'group' if plot_type == 'bar_grouped' else 'stack'
        fig = px.bar(data, x=group_by, y=selected_features[0],
                     color=selected_features[1] if len(selected_features) > 1 else None, barmode=barmode)
    elif plot_type == 'count':
        fig = px.histogram(data, x=group_by)
    elif plot_type == 'pie':
        fig = px.pie(data, names=group_by, values=selected_features[0])
    elif plot_type == 'dist':
        fig = px.histogram(data, x=selected_features[0], marginal='box', nbins=30)
    elif plot_type == 'pair':
        fig = px.scatter_matrix(data, dimensions=selected_features)
    elif plot_type == 'heatmap':
        selected_data = data[selected_features].dropna()
        corr_matrix = selected_data.corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="blues")
    elif plot_type == 'hist_kde':
        fig = px.histogram(data, x=selected_features[0], marginal='kde')
    elif plot_type == 'qq':
        qq_data = probplot(data[selected_features[0]], dist="norm")
        fig = go.Figure(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers'))
    elif plot_type == 'kde':
        fig = px.density_contour(data, x=selected_features[0],
                                 y=selected_features[1] if len(selected_features) > 1 else None)
    elif plot_type == 'reg':
        fig = px.scatter(data, x=selected_features[0], y=selected_features[1], trendline="ols")
    elif plot_type == 'box':
        fig = px.box(data, x=group_by, y=selected_features[0],
                     color=selected_features[1] if len(selected_features) > 1 else None)
    elif plot_type == 'area':
        fig = px.area(data, x=selected_features[0], y=selected_features[1])
    elif plot_type == 'violin':
        fig = px.violin(data, x=group_by, y=selected_features[0])
    elif plot_type == 'joint':
        fig = px.density_heatmap(data, x=selected_features[0], y=selected_features[1], marginal_x='scatter',
                                 marginal_y='scatter')
    elif plot_type == 'rug':
        fig = px.scatter(data, x=selected_features[0], marginal_x='rug')
    elif plot_type == '3d':
        fig = px.scatter_3d(data, x=selected_features[0], y=selected_features[1], z=selected_features[2])
    elif plot_type == 'contour':
        fig = px.density_contour(data, x=selected_features[0], y=selected_features[1])
    elif plot_type == 'cluster':
        fig = px.imshow(data[selected_features].corr(), color_continuous_scale="coolwarm")
    elif plot_type == 'hexbin':
        fig = px.density_heatmap(data, x=selected_features[0], y=selected_features[1], nbinsx=30, nbinsy=30)
    elif plot_type == 'strip':
        fig = px.strip(data, x=group_by, y=selected_features[0])
    elif plot_type == 'swarm':
        fig = px.strip(data, x=group_by, y=selected_features[0], jitter=0.5)
    else:
        fig = go.Figure()

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)




