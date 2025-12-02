import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.dashboard.components.navigation import sidebar_navigation

st.set_page_config(page_title="Hyperparameters", page_icon="🎛", layout="wide")
sidebar_navigation()

st.title("🎛 Hyperparameters: Tuning the Engine")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown(r"""
**Parameters** ($w, b$) are learned *during* training.
**Hyperparameters** are set *before* training. They control the architecture and learning process.

Examples:
*   **Capacity**: Tree Depth, Number of Neurons.
*   **Regularization**: C, Lambda, Dropout.
*   **Optimization**: Learning Rate, Batch Size.
""")

# --- 2. The Bias-Variance Tradeoff (Platinum Depth) ---
st.header("2. The Bias-Variance Tradeoff")
st.markdown(r"""
The Holy Grail of ML is **Generalization**.
We decompose the Error into three parts:

1.  **Bias (Underfitting)**: The model is too stupid. It assumes the world is simple (linear) when it's complex (curved).
    *   *Solution*: Increase Complexity (More depth, more neurons).
2.  **Variance (Overfitting)**: The model is too smart. It memorizes the noise in the training set.
    *   *Solution*: Decrease Complexity, Add Regularization, Get More Data.
3.  **Irreducible Error**: The noise inherent in the universe. (You can't fix this).
""")

st.latex(r"E[\text{Total Error}] = \text{Bias}^2 + \text{Variance} + \text{Noise}")

# --- 3. Search Strategies (Platinum Depth) ---
st.header("3. Search Strategies")
st.markdown("How do we find the best settings? It's a search problem in high-dimensional space.")

tab_grid, tab_rand, tab_bayes = st.tabs(["Grid Search", "Random Search", "Bayesian Opt"])

with tab_grid:
    st.subheader("Grid Search 🕸️")
    st.markdown(r"""
    Try **EVERY** combination.
    *   Depth: [3, 5, 10]
    *   LR: [0.01, 0.1]
    *   Total: $3 \times 2 = 6$ runs.
    *   **Pros**: Guaranteed to find the best in the grid.
    *   **Cons**: Explodes exponentially ($O(n^d)$). Impossible for many parameters.
    """)

with tab_rand:
    st.subheader("Random Search 🎲")
    st.markdown(r"""
    Try **RANDOM** combinations.
    *   Depth: RandomInt(3, 10)
    *   LR: RandomFloat(0.01, 0.1)
    *   **Pros**: Surprisingly effective. Often beats Grid Search because some parameters matter more than others.
    *   **Cons**: Might miss the absolute peak.
    """)

with tab_bayes:
    st.subheader("Bayesian Optimization 🧠")
    st.markdown(r"""
    **Smart Search**.
    1.  Try a few points.
    2.  Build a probabilistic model (Gaussian Process) of the performance surface.
    3.  Predict where the best point is likely to be.
    4.  Go there.
    *   **Pros**: Very efficient. Finds global optima with few runs.
    """)

# --- 4. Cross-Validation Strategies ---
st.header("4. Cross-Validation Strategies")
st.markdown(r"""
Never tune on the Test Set. That is cheating (Data Leakage).
Use **Cross-Validation (CV)**.

*   **K-Fold CV**: Split data into $K$ chunks. Train on $K-1$, Validate on 1. Rotate. Average the scores.
*   **Stratified K-Fold**: Ensures each chunk has the same % of wins/losses as the whole dataset. (Crucial for Imbalanced Data).
*   **Time Series Split**: Train on Past, Validate on Future. (Never shuffle time!).
""")

# --- 6. Visualization ---
st.header("6. Visualization: The Validation Curve")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    complexity = st.slider("Model Complexity", 1, 100, 50)

with col_viz:
    x = np.linspace(1, 100, 100)
    # Bias decreases
    bias = 50 * np.exp(-0.05 * x)
    # Variance increases
    variance = 0.01 * x**2
    # Total
    total = bias + variance + 10

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bias, name='Bias (Underfit)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=variance, name='Variance (Overfit)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x, y=total, name='Total Error', line=dict(color='green', width=4)))

    # Current
    curr_bias = 50 * np.exp(-0.05 * complexity)
    curr_var = 0.01 * complexity**2
    curr_tot = curr_bias + curr_var + 10

    fig.add_trace(go.Scatter(x=[complexity], y=[curr_tot], mode='markers', marker=dict(size=15, color='black'), name='Your Model'))

    fig.update_layout(title="The Bias-Variance Tradeoff", xaxis_title="Complexity", yaxis_title="Error", height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- 8. Super Summary ---
st.header("8. Super Summary 🦸")
st.info(r"""
*   **Goal**: Find the settings that generalize best.
*   **Tradeoff**: Bias vs Variance. You want the Goldilocks zone.
*   **Strategy**: Use Random Search + Stratified K-Fold CV.
*   **Rule**: If Training Error is low but Validation Error is high -> **Overfitting**.
""")
