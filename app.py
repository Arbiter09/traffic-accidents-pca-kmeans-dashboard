import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Initialize the Dash app
app = dash.Dash(__name__, title="Traffic Accidents Analysis Dashboard")
server = app.server  # For deployment

# -----------------------
# Reusable Style Dictionaries
# -----------------------
CARD_STYLE = {
    'backgroundColor': '#FFFFFF',
    'borderRadius': '8px',
    'padding': '20px',
    'margin': '10px',
    'boxShadow': '0px 2px 8px rgba(0, 0, 0, 0.1)'
}
CARD_HEADING_STYLE = {
    'textAlign': 'center',
    'color': '#2c3e50',
    'marginTop': '0px',
    'fontFamily': 'Arial, sans-serif'
}
INSTRUCTIONS_STYLE = {
    'textAlign': 'center',
    'fontStyle': 'italic',
    'color': '#7f8c8d',
    'margin': '5px 0'
}
TABLE_HEADER_STYLE = {
    'backgroundColor': 'rgb(230, 230, 230)', 
    'fontWeight': 'bold',
    'textAlign': 'center'
}
TABLE_CELL_STYLE = {
    'textAlign': 'left',
    'padding': '10px'
}

# --------------------------
# Data processing functions
# --------------------------
def prepare_data(df):
    """Prepare data for analysis by selecting and processing relevant columns."""
    if len(df.columns) >= 2:
        df = df.drop(df.columns[-2:], axis=1)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Found {len(numeric_cols)} numerical columns")
    
    df_clean = df.dropna()
    print(f"After dropping missing values: {df_clean.shape[0]} rows remain")
    
    data_numeric = df_clean[numeric_cols].copy()
    return data_numeric, numeric_cols

def perform_pca(df):
    """Perform PCA on the dataframe with proper data cleaning."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    loadings = pca.components_
    
    return pca, pca_result, explained_variance, loadings, scaled_data

def perform_kmeans(data, k_range=range(1, 11)):
    """Perform K-means clustering for a range of k values."""
    inertia = []
    models = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        models[k] = kmeans
    return inertia, models

def find_elbow_point(values):
    """Find the elbow point in a curve (for scree plot or kmeans)."""
    if len(values) <= 2:
        return 1
    
    y_diff = np.diff(values)
    
    # For scree plot (decreasing values)
    if values[0] > values[-1]:
        cum_var = np.cumsum(values)
        idx = np.argmax(cum_var >= 0.7 * cum_var[-1]) + 1
    else:
        # For kmeans (increasing values)
        y_diff2 = np.diff(y_diff)
        idx = np.argmax(np.abs(y_diff2)) + 2
    
    return min(max(1, idx), len(values) - 1)

def get_top_features(loadings, feature_names, n_components, n_top_features=4):
    """Get the top features based on squared sum of PCA loadings."""
    n_components = min(n_components, loadings.shape[0])
    squared_loadings = loadings[:n_components, :]**2
    sum_squared_loadings = np.sum(squared_loadings, axis=0)
    
    top_indices = np.argsort(sum_squared_loadings)[::-1][:n_top_features]
    
    top_features = [feature_names[i] for i in top_indices]
    top_features_scores = sum_squared_loadings[top_indices]
    
    return top_features, top_features_scores, top_indices


# --------------------
# Load and preprocess
# --------------------
try:
    file_path = "traffic_accidents_dict new.csv"
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    
    df_clean, feature_names = prepare_data(df)
    pca, pca_result, explained_variance, loadings, scaled_data = perform_pca(df_clean)
    inertia, kmeans_models = perform_kmeans(scaled_data)
    
    elbow_idx_dim = find_elbow_point(explained_variance)
    elbow_idx_k = find_elbow_point(inertia)
    
    print(f"PCA and clustering complete. Recommended dimensions: {elbow_idx_dim}, K: {elbow_idx_k}")
    
except Exception as e:
    print(f"Error processing data: {e}")
    df = pd.DataFrame()
    df_clean = pd.DataFrame()
    pca, pca_result, explained_variance, loadings, scaled_data = None, None, [], [], None
    inertia, kmeans_models = [], {}
    elbow_idx_dim, elbow_idx_k = 3, 3
    feature_names = []


# -------------
# App Layout
# -------------
app.layout = html.Div([
    html.H1(
        "Traffic Accidents Analysis Dashboard", 
        style={
            'textAlign': 'center',
            'marginTop': '20px',
            'color': '#2c3e50',
            'fontFamily': 'Arial, sans-serif'
        }
    ),

    html.P(
        [
            "Explore traffic accident data through PCA, K-Means clustering, and feature relationships. ",
            html.Br(),
            "• Click on the Scree Plot bars to select dimensionality. ",
            html.Br(),
            "• Click on the Elbow Plot points to select K. ",
            html.Br(),
            "• Move the slider in Biplot to adjust the number of feature vectors."
        ],
        style={
            'margin': '0 auto',
            'maxWidth': '800px',
            'textAlign': 'center',
            'fontStyle': 'italic',
            'color': '#7f8c8d'
        }
    ),
    
    dcc.Tabs([
        # ---------------------------
        # Tab 1: PCA & K-Means Plots
        # ---------------------------
        dcc.Tab(label="PCA & K-Means", children=[
            html.Div([
                html.Div([
                    html.H3(
                        "Scree Plot (PCA Explained Variance)",
                        style=CARD_HEADING_STYLE
                    ),
                    dcc.Graph(id='scree-plot'),
                    html.P(
                        "Click on a bar to select intrinsic dimensionality.",
                        style=INSTRUCTIONS_STYLE
                    )
                ], style={**CARD_STYLE, 'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H3(
                        "K-Means Elbow Plot",
                        style=CARD_HEADING_STYLE
                    ),
                    dcc.Graph(id='elbow-plot'),
                    html.P(
                        "Click on a point to select K value.",
                        style=INSTRUCTIONS_STYLE
                    )
                ], style={**CARD_STYLE, 'width': '48%', 'display': 'inline-block'}),
                
            ], style={'textAlign': 'center', 'margin': '20px 0'}),
        ]),
        
        # ---------------------
        # Tab 2: PCA Biplot
        # ---------------------
        dcc.Tab(label="Biplot", children=[
            html.Div([
                html.H3(
                    "PCA Biplot (PC1 vs PC2)", 
                    style=CARD_HEADING_STYLE
                ),
                
                html.Div([
                    html.Label(
                        "Number of Feature Vectors to Display:",
                        style={'fontWeight': 'bold', 'marginRight': '10px'}
                    ),
                    dcc.Slider(
                        id='vector-count',
                        min=1,
                        max=10,
                        step=1,
                        value=6,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={"placement": "bottom"}
                    )
                ], style={'width': '60%', 'margin': '20px auto'}),
                
                dcc.Graph(id='biplot'),
                html.P(
                    "Points colored by cluster. Vectors show feature directions.",
                    style=INSTRUCTIONS_STYLE
                )
            ], style={**CARD_STYLE, 'margin': '30px auto', 'maxWidth': '1000px'})
        ]),
        
        # ----------------------------
        # Tab 3: Top PCA Features
        # ----------------------------
        dcc.Tab(label="Top PCA Features", children=[
            html.Div([
                html.H3(
                    "Top Features by PCA Loading", 
                    style=CARD_HEADING_STYLE
                ),
                dash_table.DataTable(
                    id='pca-loadings-table',
                    columns=[
                        {'name': 'Feature', 'id': 'feature'},
                        {'name': 'Importance Score', 'id': 'score'},
                    ],
                    style_table={'overflowX': 'auto'},
                    style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE
                )
            ], style={**CARD_STYLE, 'margin': '30px auto', 'maxWidth': '800px'})
        ]),
        
        # --------------------------------
        # Tab 4: Scatterplot Matrix
        # --------------------------------
        dcc.Tab(label="Scatterplot Matrix", children=[
            html.Div([
                html.H3(
                    "Scatterplot Matrix (Top 4 PCA Features)", 
                    style=CARD_HEADING_STYLE
                ),
                dcc.Graph(id='scatterplot-matrix'),
                html.P(
                    "Points colored by cluster ID. Displays relationships between top features.",
                    style=INSTRUCTIONS_STYLE
                )
            ], style={**CARD_STYLE, 'margin': '30px auto', 'maxWidth': '1000px'})
        ]),
    ], style={'maxWidth': '1200px', 'margin': '40px auto'}),
    
    # Hidden stores for user selections
    dcc.Store(id='selected-dim', data=elbow_idx_dim if len(explained_variance) > 0 else 3),
    dcc.Store(id='selected-k', data=elbow_idx_k if len(inertia) > 0 else 3),
], style={'backgroundColor': '#f7f7f7', 'padding': '20px 0'})

# -----------------
# Callback: Scree Plot
# -----------------
@app.callback(
    Output('scree-plot', 'figure'),
    Output('selected-dim', 'data'),
    Input('scree-plot', 'clickData'),
    State('selected-dim', 'data')
)
def update_scree_plot(click_data, current_dim):
    if len(explained_variance) == 0:
        return {}, current_dim
    
    selected_dim = current_dim
    if click_data is not None:
        selected_dim = click_data['points'][0]['pointNumber'] + 1
    
    fig = go.Figure()
    max_components = min(20, len(explained_variance))
    
    # Define colors for bars
    default_color = 'rgba(58, 114, 176, 0.8)'  # a nice blue
    highlight_color = 'rgba(255, 127, 14, 0.8)'  # an orange highlight
    
    for i, var in enumerate(explained_variance[:max_components]):
        # Highlight the selected dimension
        color = highlight_color if (i+1) == selected_dim else default_color
        fig.add_trace(go.Bar(
            x=[i+1], 
            y=[var],
            marker_color=color,
            name=f'PC{i+1}'
        ))
    
    # Cumulative line
    cumulative = np.cumsum(explained_variance[:max_components])
    fig.add_trace(go.Scatter(
        x=list(range(1, max_components + 1)),
        y=cumulative,
        mode='lines+markers',
        marker=dict(color='rgba(214, 39, 40, 0.8)'),
        line=dict(color='rgba(214, 39, 40, 0.5)'),
        name='Cumulative'
    ))
    
    fig.update_layout(
        title='Explained Variance by Principal Component',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio',
        yaxis=dict(
            tickformat='.1%',
            range=[0, max(1, max(cumulative) * 1.1)]
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        barmode='group',
        clickmode='event+select',
        legend=dict(
            x=1.02, y=1,
            bordercolor='#ccc',
            borderwidth=1
        )
    )
    
    return fig, selected_dim

# ------------------
# Callback: Elbow Plot
# ------------------
@app.callback(
    Output('elbow-plot', 'figure'),
    Output('selected-k', 'data'),
    Input('elbow-plot', 'clickData'),
    State('selected-k', 'data')
)
def update_elbow_plot(click_data, current_k):
    if len(inertia) == 0:
        return {}, current_k
    
    selected_k = current_k
    if click_data is not None:
        selected_k = click_data['points'][0]['x']
    
    fig = go.Figure()
    k_values = list(range(1, len(inertia) + 1))
    
    default_marker = dict(size=8, color='rgba(31, 119, 180, 0.8)')
    highlight_marker = dict(size=12, color='rgba(255, 127, 14, 1)')
    
    for k, inert in zip(k_values, inertia):
        # Use highlight color for selected k
        marker_style = highlight_marker if k == selected_k else default_marker
        fig.add_trace(go.Scatter(
            x=[k], y=[inert],
            mode='markers',
            marker=marker_style,
            name=f'K={k}'
        ))
    
    # Add line connecting points
    fig.add_trace(go.Scatter(
        x=k_values, y=inertia,
        mode='lines',
        line=dict(color='rgba(31, 119, 180, 0.5)'),
        showlegend=False
    ))
    
    fig.update_layout(
        title='K-Means Elbow Plot',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Inertia (Within-Cluster Sum of Squares)',
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        ),
        clickmode='event+select',
        legend=dict(
            x=1.02, y=1,
            bordercolor='#ccc',
            borderwidth=1
        )
    )
    
    return fig, selected_k

# -------------------------------------
# Callback: PCA loadings table & Biplot
# -------------------------------------
@app.callback(
    [Output('pca-loadings-table', 'data'),
     Output('biplot', 'figure')],
    [Input('selected-dim', 'data'),
     Input('selected-k', 'data'),
     Input('vector-count', 'value')]
)
def update_pca_visualizations(selected_dim, selected_k, vector_count):
    if len(loadings) == 0 or len(feature_names) == 0:
        return [], {}
    
    # Update table
    top_features, top_scores, top_indices = get_top_features(
        loadings, feature_names, selected_dim, n_top_features=4
    )
    table_data = [
        {'feature': feature, 'score': f"{score:.4f}"}
        for feature, score in zip(top_features, top_scores)
    ]
    
    fig = go.Figure()
    marker_props = dict(size=6, opacity=0.7)
    
    # Color clusters if K>1
    if selected_k > 1 and kmeans_models and selected_k in kmeans_models:
        clusters = kmeans_models[selected_k].predict(scaled_data)
        # A few vibrant colors (can add more for higher K)
        palette = px.colors.qualitative.Safe  # or "Bold", "Pastel", etc.
        
        for cluster_id in range(selected_k):
            cluster_mask = (clusters == cluster_id)
            fig.add_trace(go.Scatter(
                x=pca_result[cluster_mask, 0],
                y=pca_result[cluster_mask, 1],
                mode='markers',
                marker={**marker_props, 'color': palette[cluster_id % len(palette)]},
                name=f'Cluster {cluster_id}'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers',
            marker=marker_props,
            name='Data Points'
        ))
    
    # Add vectors (up to vector_count)
    feature_importance = np.sqrt(loadings[0, :]**2 + loadings[1, :]**2)
    top_indices_vec = np.argsort(feature_importance)[::-1][:vector_count]
    
    x_range = np.max(pca_result[:, 0]) - np.min(pca_result[:, 0])
    y_range = np.max(pca_result[:, 1]) - np.min(pca_result[:, 1])
    axis_range = max(x_range, y_range)
    scale_factor = 0.3 * axis_range
    
    for i in top_indices_vec:
        feature = feature_names[i]
        x_end = loadings[0, i] * scale_factor
        y_end = loadings[1, i] * scale_factor
        
        # If vector is too small, skip
        if np.sqrt(x_end**2 + y_end**2) < 0.05 * scale_factor:
            continue
        
        fig.add_trace(go.Scatter(
            x=[0, x_end],
            y=[0, y_end],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ))
        fig.add_annotation(
            x=x_end, y=y_end,
            text=feature,
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255, 255, 255, 0.6)",
            borderpad=3
        )
    
    # Dotted lines at x=0 and y=0
    fig.add_shape(
        type="line",
        x0=np.min(pca_result[:, 0]) * 1.1,
        y0=0,
        x1=np.max(pca_result[:, 0]) * 1.1,
        y1=0,
        line=dict(color="gray", width=1, dash="dot"),
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=np.min(pca_result[:, 1]) * 1.1,
        x1=0,
        y1=np.max(pca_result[:, 1]) * 1.1,
        line=dict(color="gray", width=1, dash="dot"),
    )
    
    fig.update_layout(
        title='PCA Biplot (PC1 vs PC2)',
        xaxis_title='PC1',
        yaxis_title='PC2',
        legend=dict(
            x=1.05, y=1, xanchor='left', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.6)'
        ),
        margin=dict(l=50, r=50, t=60, b=50),
        height=650,
        width=900
    )
    
    # Make axes symmetrical
    max_range = max(
        abs(np.min(pca_result[:, 0])), abs(np.max(pca_result[:, 0])),
        abs(np.min(pca_result[:, 1])), abs(np.max(pca_result[:, 1]))
    )
    fig.update_xaxes(range=[-max_range*1.1, max_range*1.1])
    fig.update_yaxes(range=[-max_range*1.1, max_range*1.1])
    
    return table_data, fig

# ----------------------------------
# Callback: Scatterplot Matrix
# ----------------------------------
@app.callback(
    Output('scatterplot-matrix', 'figure'),
    [Input('selected-dim', 'data'),
     Input('selected-k', 'data')]
)
def update_scatterplot_matrix(selected_dim, selected_k):
    if df_clean.empty or len(loadings) == 0:
        return {}
    
    top_features, _, top_indices = get_top_features(
        loadings, feature_names, selected_dim, n_top_features=4
    )
    
    top_data = df_clean.iloc[:, top_indices].copy()
    
    if selected_k > 1 and kmeans_models and selected_k in kmeans_models:
        clusters = kmeans_models[selected_k].predict(scaled_data)
        top_data['Cluster'] = clusters
        color_col = 'Cluster'
    else:
        color_col = None
    
    # Jitter for categorical-like features
    categorical_features = []
    for feature in top_features:
        if len(top_data[feature].unique()) < 10:
            categorical_features.append(feature)
            if not pd.api.types.is_numeric_dtype(top_data[feature]):
                top_data[feature] = top_data[feature].astype('category').cat.codes
            top_data[f"{feature}_jittered"] = top_data[feature] + np.random.normal(
                0, 0.2, size=len(top_data)
            )
    
    plot_features = [
        f"{feat}_jittered" if feat in categorical_features else feat 
        for feat in top_features
    ]
    
    fig = px.scatter_matrix(
        top_data,
        dimensions=plot_features if categorical_features else top_features,
        color=color_col,
        labels={col: col.replace('_jittered', '') for col in plot_features},
        title="Scatterplot Matrix (Top 4 PCA Features)"
    )
    
    fig.update_layout(
        height=650,
        width=900,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            x=1.02, y=1,
            bordercolor='#ccc',
            borderwidth=1
        )
    )
    
    # Turn off diagonal histograms if desired
    fig.update_traces(diagonal_visible=False)
    
    return fig

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
