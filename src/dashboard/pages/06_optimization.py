import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.dashboard.components.navigation import sidebar_navigation

st.set_page_config(page_title="Gradient Descent", page_icon="📉", layout="wide")
sidebar_navigation()

st.title("📉 Optimization: Gradient Descent")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown("""
Machine Learning is just **Optimization**. We define a "Loss Function" $J(w)$ that measures how bad our model is, and we try to find the weights $w$ that minimize it.

**The Update Rule (Gradient Descent):**
""")
st.latex(r"w_{new} = w_{old} - \eta \cdot \nabla J(w_{old})")
st.markdown("""
*   $w$: The weights (parameters) we are tuning.
*   $J(w)$: The Loss Function (Error).
*   $\nabla J(w)$: The **Gradient**. The direction of steepest ascent (uphill).
*   $\eta$ (Eta): The **Learning Rate**. The size of the step we take downhill.
""")

# --- 2. Geometry / Structure ---
st.header("2. Geometry: The Loss Landscape")
st.markdown("""
Imagine the Loss Function as a **Mountain Range**.
*   **High Altitude**: High Error (Bad Model).
*   **Sea Level**: Zero Error (Perfect Model).
*   **Coordinates**: The weights $w_1, w_2$.

We are a blind hiker dropped somewhere on the mountain. We feel the slope under our feet and take a step **down**.
""")

# --- 3. Constraints / Objective / Loss ---
st.header("3. The Gradient Vector")
st.markdown("""
The Gradient is a vector of partial derivatives. It points in the direction where the function increases the fastest.
""")
st.latex(r"\nabla J(w) = \begin{bmatrix} \frac{\partial J}{\partial w_1} \\ \frac{\partial J}{\partial w_2} \\ \vdots \end{bmatrix}")
st.markdown("""
*   If $\frac{\partial J}{\partial w_1} > 0$: Increasing $w_1$ increases Loss. So we should **decrease** $w_1$.
*   If $\frac{\partial J}{\partial w_1} < 0$: Increasing $w_1$ decreases Loss. So we should **increase** $w_1$.
*   The minus sign in the update rule ($- \eta \nabla J$) handles this logic automatically.
""")

# --- 4. Deeper Components (Convexity) ---
st.header("4. Convexity & Local Minima")
st.markdown("""
*   **Convex Function**: Shaped like a perfect bowl. Has only **one** minimum (Global Minimum). Gradient Descent is guaranteed to find it. (e.g., Linear Regression, Logistic Regression).
*   **Non-Convex Function**: Wavy, with many valleys. Has many **Local Minima**. Gradient Descent might get stuck in a suboptimal valley. (e.g., Neural Networks).
""")

# --- 6. Visualization ---
st.header("6. Visualization: The 3D Surface")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    lr = st.slider("Learning Rate", 0.01, 1.2, 0.1)
    steps = st.slider("Steps", 1, 50, 20)
    start_x = st.slider("Start X", -9.0, 9.0, 8.0)

with col_viz:
    # Surface Data
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple Bowl

    # Path Calculation
    path_x = [start_x]
    path_y = [0] # Keep y constant for simplicity in 2D path, or vary it
    path_z = [start_x**2]

    curr_x = start_x
    curr_y = 0

    for _ in range(steps):
        grad_x = 2 * curr_x
        grad_y = 2 * curr_y

        curr_x = curr_x - lr * grad_x
        curr_y = curr_y - lr * grad_y

        path_x.append(curr_x)
        path_y.append(curr_y)
        path_z.append(curr_x**2 + curr_y**2)

    fig = go.Figure()

    # Surface
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, showscale=False))

    # Path
    fig.add_trace(go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='markers+lines',
                               marker=dict(size=5, color='red'), line=dict(color='red', width=5), name='Path'))

    fig.update_layout(title="Gradient Descent on J(w) = w1^2 + w2^2",
                      scene=dict(xaxis_title='w1', yaxis_title='w2', zaxis_title='Loss'), height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- 7. Hyperparameters ---
st.header("7. Hyperparameters & Behavior")
st.markdown("""
*   **Learning Rate ($\eta$)**:
    *   **Too Small**: Tiny steps. Takes forever to reach the bottom.
    *   **Too Large**: Huge steps. Might overshoot the bottom and diverge (explode).
    *   **Just Right**: Converges quickly and stably.
""")

# --- 8. Super Summary ---
st.header("8. Super Summary 🦸")
st.info("""
*   **Goal**: Find weights $w$ that minimize Loss $J(w)$.
*   **Math**: $w \leftarrow w - \eta \nabla J$.
*   **Key Insight**: Follow the slope downhill.
*   **Knobs**: Learning Rate is the most critical parameter.
""")
