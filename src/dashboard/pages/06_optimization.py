import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.dashboard.components.navigation import sidebar_navigation

st.set_page_config(page_title="Gradient Descent", page_icon="📉", layout="wide")
sidebar_navigation()

st.title("📉 Optimization: Gradient Descent")

# 1. Intuition
st.header("1. Intuition: Hiking Down a Mountain")
st.markdown("""
### The Analogy
Imagine you are dropped on a random spot on a massive mountain range at night. It is pitch black; you cannot see the bottom. Your goal is to reach the lowest point (the valley) because that's where the village is.

**How do you do it?**
1.  **Feel the slope** of the ground under your feet.
2.  Identify which direction goes **downhill**.
3.  Take a **small step** in that direction.
4.  Repeat until the ground is flat (slope = 0).

**In Machine Learning:**
*   **The Mountain**: The **Loss Function** (Error). We want to minimize it.
*   **Your Position**: The **Model Weights** ($w$).
*   **The Slope**: The **Gradient** ($\nabla J$).
*   **Step Size**: The **Learning Rate** ($\eta$).
""")

st.markdown("---")

# 2. Math
st.header("2. The Math: Following the Gradient")
st.markdown("We want to find the weights $w$ that minimize the Loss $J(w)$.")

st.subheader("The Update Rule")
st.latex(r"w_{new} = w_{old} - \eta \cdot \frac{\partial J}{\partial w}")

st.markdown("""
*   $\frac{\partial J}{\partial w}$: The **Gradient**. It points in the direction of steepest ascent (uphill).
*   $-\frac{\partial J}{\partial w}$: Points in the direction of steepest descent (downhill).
*   $\eta$ (Eta): The **Learning Rate**. Controls how big of a step we take.
""")

st.subheader("Derivation Example: Simple Quadratic")
st.markdown("Let's minimize a simple function: $J(w) = w^2$.")
st.markdown("We know the minimum is at $w=0$. Let's see if the math finds it.")

st.markdown("**Step 1: Find the Derivative (Slope)**")
st.latex(r"\frac{\partial J}{\partial w} = \frac{d}{dw}(w^2) = 2w")

st.markdown("**Step 2: Plug into Update Rule**")
st.latex(r"w_{new} = w_{old} - \eta \cdot (2w_{old})")

st.markdown("---")

# 3. Worked Example
st.header("3. Step-by-Step Worked Example")
st.markdown("""
*   **Start Position**: $w_0 = 3$
*   **Learning Rate**: $\eta = 0.1$
*   **Gradient**: $\nabla J = 2w$
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Iteration 1")
    st.markdown("1. Current $w = 3$.")
    st.markdown("2. Gradient = $2(3) = 6$. (Slope is steep positive).")
    st.markdown("3. Step = $0.1 \times 6 = 0.6$.")
    st.markdown("4. New $w = 3 - 0.6 = 2.4$.")
    st.success("Moved from 3 to 2.4 (Closer to 0!)")

with col2:
    st.markdown("#### Iteration 2")
    st.markdown("1. Current $w = 2.4$.")
    st.markdown("2. Gradient = $2(2.4) = 4.8$. (Slope is less steep).")
    st.markdown("3. Step = $0.1 \times 4.8 = 0.48$.")
    st.markdown("4. New $w = 2.4 - 0.48 = 1.92$.")
    st.success("Moved from 2.4 to 1.92 (Even closer!)")

st.markdown("Notice how the steps get smaller as we get closer to the bottom? That's because the slope (gradient) gets smaller.")

st.markdown("---")

# 4. Interactive Viz
st.header("4. Interactive Visualization")
st.markdown("Play with the Learning Rate. See what happens if it's too high or too low.")

col1, col2 = st.columns([1, 3])

with col1:
    learning_rate = st.slider("Learning Rate ($\eta$)", 0.01, 1.2, 0.1, 0.01)
    start_w = st.slider("Starting Position ($w$)", -10.0, 10.0, 8.0)
    steps = st.slider("Number of Steps", 1, 50, 10)

with col2:
    # Function J(w) = w^2
    w_range = np.linspace(-12, 12, 100)
    J_range = w_range ** 2

    # Gradient Descent Simulation
    path_w = [start_w]
    path_J = [start_w ** 2]

    current_w = start_w
    for _ in range(steps):
        gradient = 2 * current_w  # Derivative of w^2 is 2w
        current_w = current_w - learning_rate * gradient
        path_w.append(current_w)
        path_J.append(current_w ** 2)

    fig = go.Figure()

    # Plot Function
    fig.add_trace(go.Scatter(x=w_range, y=J_range, mode='lines', name='Loss Function J(w)'))

    # Plot Path
    fig.add_trace(go.Scatter(
        x=path_w, y=path_J,
        mode='markers+lines',
        name='Gradient Descent Path',
        marker=dict(size=10, color='red'),
        line=dict(dash='dot', color='red')
    ))

    fig.update_layout(title=f"Gradient Descent (Final w: {current_w:.4f})", xaxis_title="Weight w", yaxis_title="Loss J(w)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    if learning_rate > 1.0:
        st.error("⚠️ **Overshooting!** The Learning Rate is too high. The ball is jumping across the valley and moving UP the other side.")
    elif learning_rate < 0.05:
        st.warning("⚠️ **Too Slow!** The Learning Rate is tiny. It will take forever to reach the bottom.")
    else:
        st.success("✅ **Just Right.** The ball descends smoothly and slows down as it reaches the bottom.")

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Back to Playground", icon="🎮")
