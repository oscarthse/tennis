import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.dashboard.components.navigation import sidebar_navigation

st.set_page_config(page_title="Gradient Descent", page_icon="📉", layout="wide")
sidebar_navigation()

st.title("📉 Optimization: The Engine of Learning")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown(r"""
Machine Learning is fundamentally an **Optimization Problem**.
We define a **Loss Function** $J(w)$ that measures "Badness".
Our goal is to find the weights $w^*$ that minimize this badness.

**The Algorithm: Gradient Descent**
Imagine you are a blind hiker on a mountain. You want to reach the bottom (Minimum Loss).
1.  Feel the slope under your feet ($\nabla J$).
2.  Take a step downhill.
3.  Repeat.

**The Update Rule:**
""")
st.latex(r"w_{t+1} = w_t - \eta \cdot \nabla J(w_t)")
st.markdown(r"""
*   $w$: The weights (parameters).
*   $\nabla J$: The **Gradient** (Steepest Ascent).
*   $\eta$ (Eta): The **Learning Rate** (Step Size).
""")

# --- 2. Geometry / Structure ---
st.header("2. Geometry: The Loss Landscape")
st.markdown(r"""
*   **Convex**: A perfect bowl. Has one global minimum. (e.g., Linear/Logistic Regression). Easy to solve.
*   **Non-Convex**: A rugged landscape with many peaks and valleys. Has many **Local Minima**. (e.g., Neural Networks). Hard to solve.

**The Gradient Vector**:
The gradient is a vector of partial derivatives. It points **Uphill**.
""")
st.latex(r"\nabla J(w) = \left[ \frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2}, \dots \right]^T")

st.markdown("**Why the Minus Sign?**")
st.latex(r"\text{If } \frac{\partial J}{\partial w} > 0 \implies \text{Slope is Positive (Uphill)} \implies \text{We must go Left (Decrease } w\text{)}")
st.latex(r"\text{If } \frac{\partial J}{\partial w} < 0 \implies \text{Slope is Negative (Downhill)} \implies \text{We must go Right (Increase } w\text{)}")

# --- 3. Advanced Optimizers (Platinum Depth) ---
st.header("3. Advanced Optimizers: Beyond Vanilla GD")
st.markdown("Vanilla Gradient Descent has problems. It gets stuck in flat areas (plateaus) and oscillates in ravines.")

tab_mom, tab_adam, tab_sgd = st.tabs(["Momentum", "Adam", "Batch vs SGD"])

with tab_mom:
    st.subheader("Momentum: The Heavy Ball 🎳")
    st.markdown(r"""
    Imagine a heavy ball rolling down the hill. It gains **Momentum**.
    *   If the gradient is small (flat), the ball keeps rolling because of its speed.
    *   This helps it power through small bumps and plateaus.

    **The Math:**
    We keep a "Velocity" vector $v$.
    """)
    st.latex(r"v_{t+1} = \gamma v_t + \eta \nabla J(w_t)")
    st.latex(r"w_{t+1} = w_t - v_{t+1}")
    st.markdown(r"*   $\gamma$ (Gamma): Friction (usually 0.9). Retains 90% of previous speed.")

with tab_adam:
    st.subheader("Adam: The Smart Hiker 🧠")
    st.markdown(r"""
    **Ada**ptive **M**oment Estimation.
    *   It adapts the learning rate for *each parameter* individually.
    *   **Sparse Features**: Rarely updated parameters get huge steps.
    *   **Frequent Features**: Frequently updated parameters get small steps.

    It combines Momentum (First Moment) and RMSProp (Second Moment).
    It is the **Default** optimizer for Deep Learning today.
    """)

with tab_sgd:
    st.subheader("Batch vs. Stochastic vs. Mini-Batch")
    st.markdown(r"""
    How much data do we look at before taking a step?

    1.  **Batch GD**: Look at **ALL** data. (Precise, but Slow).
        *   "I read the entire map before taking one step."
    2.  **Stochastic GD (SGD)**: Look at **ONE** sample. (Fast, but Noisy/Drunk).
        *   "I look at one tree, take a step. Look at another tree, take a step."
    3.  **Mini-Batch GD**: Look at **32 or 64** samples. (Best of both worlds).
        *   "I look at a small patch of terrain, then move."
    """)

# --- 6. Visualization ---
st.header("6. Visualization: The 3D Surface")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    lr = st.slider("Learning Rate", 0.01, 1.2, 0.1)
    steps = st.slider("Steps", 1, 100, 50)
    optimizer = st.selectbox("Optimizer", ["GD", "Momentum"])
    momentum = st.slider("Momentum (Gamma)", 0.0, 0.99, 0.9) if optimizer == "Momentum" else 0.0

with col_viz:
    # Surface Data
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple Bowl

    # Path Calculation
    path_x = [8.0]
    path_y = [0.0]
    path_z = [64.0]

    curr_x = 8.0
    curr_y = 0.0
    vel_x = 0.0
    vel_y = 0.0

    for _ in range(steps):
        grad_x = 2 * curr_x
        grad_y = 2 * curr_y # + 5 * np.sin(curr_y) # Add some ruggedness? No, keep simple for demo

        if optimizer == "Momentum":
            vel_x = momentum * vel_x + lr * grad_x
            vel_y = momentum * vel_y + lr * grad_y
            curr_x = curr_x - vel_x
            curr_y = curr_y - vel_y
        else:
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
                               marker=dict(size=4, color='red'), line=dict(color='red', width=4), name='Path'))

    fig.update_layout(title=f"Optimization Path ({optimizer})",
                      scene=dict(xaxis_title='w1', yaxis_title='w2', zaxis_title='Loss'), height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- 8. Super Summary ---
st.header("8. Super Summary 🦸")
st.info(r"""
*   **Goal**: Find the bottom of the valley (Min Loss).
*   **Gradient**: The compass pointing Uphill. We go opposite.
*   **Learning Rate**: Step size. Too big = Explode. Too small = Slow.
*   **Momentum**: Helps plow through flat areas and dampen oscillations.
*   **SGD**: Fast, noisy updates. **Batch**: Slow, precise updates.
""")
