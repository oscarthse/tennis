import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.dashboard.components.navigation import sidebar_navigation

st.set_page_config(page_title="Hyperparameters", page_icon="🎛", layout="wide")
sidebar_navigation()

st.title("🎛 Hyperparameters & Tuning")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown("""
**Parameters** ($w, b$) are learned by the model from data.
**Hyperparameters** are settings YOU choose *before* training. They control **how** the model learns.

Examples:
*   Learning Rate ($\eta$)
*   Regularization Strength ($C$)
*   Tree Depth
*   Number of Neighbors ($k$)
""")

# --- 2. Geometry / Structure ---
st.header("2. Geometry: The Bias-Variance Tradeoff")
st.markdown("""
This is the fundamental problem of Machine Learning.

*   **Underfitting (High Bias)**: The model is too simple. It ignores the data structure. (e.g., a straight line for a curved boundary).
*   **Overfitting (High Variance)**: The model is too complex. It memorizes the noise in the training data. (e.g., a squiggly line touching every point).
*   **Sweet Spot**: The perfect balance where Validation Error is minimized.
""")

# --- 3. Constraints / Objective / Loss ---
st.header("3. The Decomposition")
st.markdown("The Expected Error can be mathematically decomposed:")
st.latex(r"E[\text{Error}] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}")
st.markdown("""
*   **Bias**: Error from erroneous assumptions (e.g., assuming linear).
*   **Variance**: Error from sensitivity to small fluctuations in the training set.
*   **Irreducible Error**: Noise in the real world (can't be fixed).
""")

# --- 6. Visualization ---
st.header("6. Visualization: The Validation Curve")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    complexity = st.slider("Model Complexity", 1, 100, 50)

with col_viz:
    # Synthetic Curves
    x = np.linspace(1, 100, 100)

    # Bias decreases with complexity
    bias = 100 * np.exp(-0.05 * x)

    # Variance increases with complexity
    variance = 0.01 * x**2

    # Total Error
    total_error = bias + variance + 10 # +10 irreducible

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bias, name='Bias^2 (Underfitting)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=variance, name='Variance (Overfitting)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x, y=total_error, name='Total Error', line=dict(color='green', width=4)))

    # Current Point
    curr_bias = 100 * np.exp(-0.05 * complexity)
    curr_var = 0.01 * complexity**2
    curr_tot = curr_bias + curr_var + 10

    fig.add_trace(go.Scatter(x=[complexity], y=[curr_tot], mode='markers', marker=dict(size=15, color='black'), name='Your Model'))

    fig.update_layout(title="Bias-Variance Tradeoff", xaxis_title="Model Complexity", yaxis_title="Error", height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- 8. Super Summary ---
st.header("8. Super Summary 🦸")
st.info("""
*   **Goal**: Find the "Sweet Spot" of complexity.
*   **Math**: Error = Bias² + Variance + Noise.
*   **Key Insight**: Simple models underfit. Complex models overfit.
*   **Knobs**: Grid Search or Random Search to find the best settings.
""")
