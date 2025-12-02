import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.model_cards import render_model_card
from src.dashboard.components.toy_datasets import generate_moons, generate_linear
from src.dashboard.components.mermaid import render_mermaid

st.set_page_config(page_title="Logistic Regression", page_icon="📈", layout="wide")
sidebar_navigation()

st.title("📈 Logistic Regression: The Foundation of Probability")

# --- LAYER 1: Super Simple Intuition ---
st.header("1. Intuition: The Dimmer Switch 💡")
st.markdown("""
Imagine a light switch.
*   **Classification (Hard)**: It's either ON or OFF. No middle ground.
*   **Regression (Linear)**: It's a dimmer switch that can go from 0 to infinity (which doesn't make sense for a light).
*   **Logistic Regression**: It's a **smart dimmer** that knows the limits.
    *   It can be fully OFF (0).
    *   It can be fully ON (1).
    *   But it can also be "80% Bright" (Probability = 0.8).

It takes any input (like "how hard you push the button") and squashes it into a safe range between 0 and 1.
""")

st.markdown("---")

# --- LAYER 2: Real-World Analogy ---
st.header("2. Analogy: The Tennis Rank Difference 🎾")
st.markdown("""
We want to predict if **Nadal** wins based on **Rank Difference** (Player Rank - Opponent Rank).

*   **Scenario A (Huge Advantage)**: Nadal (2) vs. Unknown (100). Diff = -98.
    *   Intuition: "He definitely wins." -> Probability $\approx$ 1.0.
*   **Scenario B (Even Match)**: Nadal (2) vs. Djokovic (1). Diff = +1.
    *   Intuition: "Could go either way." -> Probability $\approx$ 0.5.
*   **Scenario C (Huge Disadvantage)**: Nadal (100) vs. Djokovic (1). Diff = +99.
    *   Intuition: "He definitely loses." -> Probability $\approx$ 0.0.

We need a mathematical function that behaves exactly like this intuition:
*   Input $-\infty \to$ Output 0
*   Input $0 \to$ Output 0.5
*   Input $+\infty \to$ Output 1
""")

st.markdown("---")

# --- LAYER 3: Structural Explanation ---
st.header("3. Structure: How it Works Mechanically ⚙️")
st.markdown("Logistic Regression is a machine with **three parts**:")

render_mermaid("""
graph LR
    Input["Inputs (x) <br> Rank, Points"] -->|Dot Product| Score["Linear Score (z) <br> -inf to +inf"]
    Score -->|Sigmoid Function| Prob["Probability (p) <br> 0 to 1"]
    Prob -->|Threshold 0.5| Class["Prediction (y_hat) <br> Win/Lose"]

    style Score fill:#fff3e0
    style Prob fill:#e8f5e9
""", height=250)

st.markdown("""
1.  **The Linear Score ($z$)**: Combines all features into a single number.
    *   $z = w_1 \cdot \text{Rank} + w_2 \cdot \text{Points} + b$
2.  **The Activation ($\sigma$)**: The "Squashing Function" (Sigmoid).
    *   $p = \frac{1}{1 + e^{-z}}$
3.  **The Decision**: If $p > 0.5$, predict Win.
""")

st.markdown("---")

# --- LAYER 4: Step-by-Step Breakdown ---
st.header("4. Step-by-Step Prediction 👣")
st.markdown("Let's trace a single prediction manually.")

st.markdown("""
**The Match**:
*   **Rank Diff ($x_1$)**: 10 (Player is worse)
*   **Points Diff ($x_2$)**: -500 (Player has fewer points)

**The Model (Trained Weights)**:
*   $w_1 = -0.1$ (Rank hurts)
*   $w_2 = 0.002$ (Points help)
*   $b = 0.5$ (Base bias)
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Step 1: Calculate Weighted Sum ($z$)**")
    st.latex(r"z = (w_1 \cdot x_1) + (w_2 \cdot x_2) + b")
    st.latex(r"z = (-0.1 \cdot 10) + (0.002 \cdot -500) + 0.5")
    st.latex(r"z = -1.0 - 1.0 + 0.5 = -1.5")
    st.info("The score is negative, so we expect a Loss.")

with col2:
    st.markdown("**Step 2: Apply Sigmoid ($p$)**")
    st.latex(r"p = \frac{1}{1 + e^{-(-1.5)}} = \frac{1}{1 + e^{1.5}}")
    st.latex(r"p = \frac{1}{1 + 4.48} \approx 0.18")
    st.success("Probability = 18%. Prediction: LOSE.")

st.markdown("---")

# --- LAYER 5: Full Math ---
st.header("5. The Math: Deep Dive 🧮")

st.subheader("A. The Log-Odds (Where does Sigmoid come from?)")
st.markdown("Why do we use that weird $1/(1+e^{-z})$ function?")
st.markdown("It comes from the concept of **Odds**.")
st.latex(r"\text{Odds} = \frac{P(\text{Win})}{P(\text{Lose})} = \frac{p}{1-p}")

st.markdown("We want to model the **Log-Odds** as a linear line:")
st.latex(r"\ln\left(\frac{p}{1-p}\right) = z = w^T x + b")

st.markdown("If we solve this equation for $p$, we get the Sigmoid!")
st.latex(r"\frac{p}{1-p} = e^z \implies p = (1-p)e^z \implies p(1+e^z) = e^z \implies p = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}}")

st.subheader("B. The Loss Function (Maximum Likelihood)")
st.markdown("We don't use Least Squares (MSE). We use **Log Loss** (Cross-Entropy).")
st.markdown("We want to maximize the likelihood of the correct labels:")
st.latex(r"L(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]")

st.subheader("C. The Gradient Derivation (Chain Rule)")
st.markdown("To train the model, we need $\\frac{\partial L}{\partial w}$. Let's derive it properly.")
st.latex(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial w}")

col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    st.markdown("1. Loss derivative")
    st.latex(r"\frac{p-y}{p(1-p)}")
with col_d2:
    st.markdown("2. Sigmoid derivative")
    st.latex(r"p(1-p)")
with col_d3:
    st.markdown("3. Linear derivative")
    st.latex(r"x")

st.markdown("Multiplying them cancels the denominator:")
st.latex(r"\nabla L = (p - y) \cdot x")
st.success("This is the 'Error' times the 'Input'. Beautifully simple.")

st.markdown("---")

# --- LAYER 6: Diagrams ---
st.header("6. Visualization: The S-Curve")

z_vals = np.linspace(-6, 6, 100)
p_vals = 1 / (1 + np.exp(-z_vals))

fig_sig = go.Figure()
fig_sig.add_trace(go.Scatter(x=z_vals, y=p_vals, mode='lines', name='Sigmoid'))
fig_sig.add_shape(type="line", x0=-6, y0=0.5, x1=6, y1=0.5, line=dict(color="red", dash="dash"))
fig_sig.add_annotation(x=0, y=0.5, text="Decision Threshold (0.5)", showarrow=True, arrowhead=1)
fig_sig.update_layout(title="The Sigmoid Function", xaxis_title="Score (z)", yaxis_title="Probability (p)", height=400)
st.plotly_chart(fig_sig, use_container_width=True)

st.markdown("---")

# --- LAYER 7: Micro-Examples ---
st.header("7. Micro-Examples 🧪")

st.markdown("**Example 1: The 'Sure Thing'**")
st.markdown("*   $z = 5.0$")
st.markdown("*   $p = 1 / (1 + e^{-5}) \approx 0.993$")
st.markdown("*   **Result**: 99.3% Confidence.")

st.markdown("**Example 2: The 'Toss-up'**")
st.markdown("*   $z = 0.0$")
st.markdown("*   $p = 1 / (1 + 1) = 0.5$")
st.markdown("*   **Result**: 50% Confidence.")

st.markdown("---")

# --- LAYER 8: FAQ ---
st.header("8. FAQ 🙋")
with st.expander("Q: Why not just use Linear Regression for classification?"):
    st.markdown("""
    Linear Regression outputs numbers like -10 or +50.
    1.  It violates probability rules (must be 0-1).
    2.  It is sensitive to outliers (a point at $x=1000$ pulls the line too much).
    """)
with st.expander("Q: What does the weight 'w' actually mean?"):
    st.markdown("""
    It is the **Feature Importance**.
    *   Large positive $w$: Feature strongly increases win chance.
    *   Large negative $w$: Feature strongly decreases win chance.
    *   Zero $w$: Feature is irrelevant.
    """)

st.markdown("---")

# --- LAYER 9: Exercises ---
st.header("9. Exercises 📝")
st.info("""
1.  **Calculate**: If $z = 2.0$, what is $p$? (Hint: $e^{-2} \approx 0.135$)
2.  **Derive**: Prove that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
3.  **Think**: If we double all weights $w$, what happens to the S-curve? (It gets steeper!)
""")

st.markdown("---")

# --- Interactive Playground ---
st.header("10. Interactive Playground")
st.markdown("Train a Logistic Regression model on a toy dataset.")

col1, col2 = st.columns([1, 3])
with col1:
    dataset_type = st.selectbox("Dataset", ["Linear", "Moons"])
    noise = st.slider("Noise", 0.0, 1.0, 0.2)
    C_param = st.slider("Regularization (C)", 0.01, 10.0, 1.0)

with col2:
    if dataset_type == "Linear":
        X, y = generate_linear(noise=noise)
    else:
        X, y = generate_moons(noise=noise)

    clf = LogisticRegression(C=C_param)
    clf.fit(X, y)

    # Plot Decision Boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z, colorscale='RdBu', opacity=0.8, showscale=False))
    fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', name='Class 0', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', name='Class 1', marker=dict(color='blue')))
    fig.update_layout(title=f"Decision Boundary (Acc: {clf.score(X, y):.2f})", height=500)
    st.plotly_chart(fig, use_container_width=True)

st.page_link("pages/02_model_playground.py", label="🎮 Go to Playground", icon="🎮")
