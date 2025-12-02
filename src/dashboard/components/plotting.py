import plotly.graph_objects as go
import plotly.express as px
from sklearn.calibration import calibration_curve
import pandas as pd

def plot_gauge(probability, player_name, is_winner):
    """Create a gauge chart for win probability."""
    color = '#2ecc71' if is_winner else '#e74c3c'

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=probability * 100,
        title={'text': f'{player_name} Win Probability', 'font': {'size': 20}},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#ecf0f1'},
                {'range': [50, 100], 'color': '#d5dbdb'}
            ],
            'threshold': {'line': {'color': 'black', 'width': 2}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_confusion_matrix(cm, labels=['Player 2', 'Player 1']):
    """Create a heatmap for the confusion matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Pred {l}' for l in labels],
        y=[f'Actual {l}' for l in labels],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={'size': 20}
    ))
    fig.update_layout(height=400, title='Confusion Matrix')
    return fig

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """Create a calibration curve plot."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode='lines+markers',
        name='Model',
        line=dict(color='#3498db')
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfectly Calibrated',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title='Calibration Curve',
        xaxis_title='Predicted Probability',
        yaxis_title='True Probability',
        height=400,
        legend=dict(x=0.01, y=0.99)
    )
    return fig

def plot_metric_comparison(df, metric, title, color_scale='Blues'):
    """Create a bar chart comparing models on a specific metric."""
    fig = px.bar(
        df,
        x='Model',
        y=metric,
        title=title,
        color=metric,
        color_continuous_scale=color_scale,
        text_auto='.3f'
    )
    fig.update_layout(showlegend=False, height=400)
    return fig

def plot_grouped_accuracy(df, group_col, title):
    """Create a bar chart for accuracy grouped by a column."""
    fig = px.bar(
        df,
        x=group_col,
        y='Accuracy',
        title=title,
        color='Accuracy',
        color_continuous_scale='Viridis',
        text_auto='.2f'
    )
    fig.update_layout(height=400)
    return fig
