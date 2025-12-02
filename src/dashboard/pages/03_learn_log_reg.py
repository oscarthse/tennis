import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.model_cards import render_model_card
from src.dashboard.components.math_blocks import get_logistic_regression_latex
from src.dashboard.components.toy_datasets import generate_moons, generate_linear

st.set_page_config(page_title="Logistic Regression", page_icon="📈", layout="wide")
sidebar_navigation()

st.title("📈 Logistic Regression: The Foundation of Probability")

# 1. TL;DR
render_model_card(
    title="Logistic Regression",
    description="The foundational building block of modern AI. It predicts the **probability** of an event (Win/Loss) by fitting a linear boundary and squashing the output between 0 and 1 using the Sigmoid function.",
    pros=["Interpretable (weights = feature importance)", "Fast to train", "Outputs probabilities (confidence)"],
    cons=["Can only solve linear problems (unless feature engineering)", "Sensitive to outliers"]
)

st.markdown("---")

# 2. Intuition
st.header("1. Intuition: From Lines to Probabilities")
st.markdown("""
### The Tennis Analogy
Imagine you are trying to predict if **Nadal** will win a match based on a single number: the **Rank Difference** between him and his opponent.

*   **Scenario A**: Nadal (Rank 2) vs. Unknown (Rank 100). Difference = +98. You are **very confident** Nadal wins.
*   **Scenario B**: Nadal (Rank 2) vs. Djokovic (Rank 1). Difference = -1. You are **uncertain** (maybe 50/50).
*   **Scenario C**: Nadal (Rank 100) vs. Djokovic (Rank 1). Difference = -99. You are **very confident** Nadal loses.

If we plotted this, we wouldn't want a straight line that goes to infinity (you can't have 150% probability). We want an **S-curve** that starts at 0 (Lose), rises steeply through 0.5 (Uncertain), and flattens out at 1 (Win).

**That S-curve is the Sigmoid Function.**

Logistic Regression does two things:
1.  **The Score ($z$)**: Calculates a weighted score based on inputs (Rank, Points, Surface).
    *   $z = w \cdot \text{RankDiff} + b$
2.  **The Probability ($p$)**: Squashes that score into a probability using the Sigmoid.
    *   $p = \text{Sigmoid}(z)$
""")

st.markdown("---")

# 3. The Math
st.header("2. The Math: Rigorous Derivation")

st.subheader("Step 1: The Linear Model (The Score)")
st.markdown("First, we compute a linear combination of the input features $x$ and weights $w$. This is exactly like Linear Regression.")
st.latex(r"z = w^\top x + b = w_1 x_1 + w_2 x_2 + \dots + b")
st.markdown("*   $z$: The 'Log-Odds' or 'Logit'. A raw score from $-\infty$ to $+\infty$.*")

st.subheader("Step 2: The Sigmoid Activation")
st.markdown("We need to map $z$ to the range $[0, 1]$. We use the Sigmoid function $\sigma(z)$:")
st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")

st.markdown("**Why this function?**")
st.markdown("It comes from the definition of **Odds**. If $p$ is the probability of winning:")
st.latex(r"\text{Odds} = \frac{p}{1-p} = e^z \implies \ln(\text{Odds}) = z")
st.markdown("Solving for $p$ gives us the Sigmoid function.")

st.subheader("Step 3: The Loss Function (Log Loss)")
st.markdown("How do we find the best weights $w$? We can't use Mean Squared Error (MSE) because the output is a probability. Instead, we use **Maximum Likelihood Estimation**.")
st.markdown("We want to maximize the probability assigned to the correct label $y$.")

st.latex(r"L(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]")

st.markdown("""
*   If $y=1$ (Win): We want $\hat{y}$ to be close to 1. The loss is $-\log(\hat{y})$.
*   If $y=0$ (Loss): We want $\hat{y}$ to be close to 0. The loss is $-\log(1 - \hat{y})$.
""")

st.subheader("Step 4: Gradient Descent")
st.markdown("To minimize the loss, we need the gradient $\\nabla L$. Many courses just give you the formula. **We will derive it.**")

st.markdown("We use the **Chain Rule** from Calculus:")
st.latex(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial w}")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown("**1. Loss w.r.t Probability**")
    st.latex(r"\frac{\partial L}{\partial p} = \frac{p - y}{p(1-p)}")
    st.caption("(Derived from Log Loss)")

with col_d2:
    st.markdown("**2. Probability w.r.t Score**")
    st.latex(r"\frac{\partial p}{\partial z} = p(1-p)")
    st.caption("(Derivative of Sigmoid)")

with col_d3:
    st.markdown("**3. Score w.r.t Weights**")
    st.latex(r"\frac{\partial z}{\partial w} = x")
    st.caption("(Derivative of $wx+b$)")

st.markdown("Now, multiply them all together. The magic happens in the middle:")
st.latex(r"\frac{\partial L}{\partial w} = \left( \frac{p - y}{p(1-p)} \right) \cdot (p(1-p)) \cdot x")
st.markdown("The $p(1-p)$ terms **cancel out** perfectly!")

st.success("Final Gradient Formula:")
st.latex(r"\frac{\partial L}{\partial w} = (p - y) \cdot x")

st.markdown("""
**Interpretation:**
*   $(p - y)$: The **Error**. (Prediction - Actual).
*   $x$: The **Input**.
*   The gradient tells us: "If the error is positive, decrease the weight for this input."
""")

st.markdown("---")

# 4. Worked Example
st.header("3. Step-by-Step Worked Example")
st.markdown("Let's calculate the probability for a single match manually.")

st.markdown("### The Match")
st.markdown("*   **Feature ($x$)**: Rank Difference (Player - Opponent). Let's say $x = 10$ (Player is worse).")
st.markdown("*   **Weight ($w$)**: $-0.1$ (Negative because higher rank diff means lower chance of winning).")
st.markdown("*   **Bias ($b$)**: $0.5$ (Base advantage).")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Step 1: Calculate Logit ($z$)")
    st.latex(r"z = w \cdot x + b")
    st.latex(r"z = (-0.1 \cdot 10) + 0.5")
    st.latex(r"z = -1.0 + 0.5 = -0.5")

with col2:
    st.markdown("#### Step 2: Calculate Probability ($p$)")
    st.latex(r"p = \frac{1}{1 + e^{-z}}")
    st.latex(r"p = \frac{1}{1 + e^{-(-0.5)}} = \frac{1}{1 + e^{0.5}}")
    st.latex(r"p = \frac{1}{1 + 1.648} \approx 0.377")

st.success("Result: The model predicts a **37.7%** chance of winning.")

st.markdown("#### Step 3: Calculate Loss")
st.markdown("Suppose the player **actually won** ($y=1$). How bad was our prediction?")
st.latex(r"Loss = -[1 \cdot \log(0.377) + 0 \cdot \log(1-0.377)]")
st.latex(r"Loss = -\log(0.377) \approx 0.975")
st.markdown("The loss is high because we predicted < 50% but they won.")

st.markdown("---")

# 5. Code
st.header("4. The Code")
st.code("""
from sklearn.linear_model import LogisticRegression

# 1. Initialize the model
# C is the inverse of regularization strength (like 1/lambda)
model = LogisticRegression(C=1.0, solver='lbfgs')

# 2. Train (Fit)
# X_train: [n_samples, n_features], y_train: [n_samples]
model.fit(X_train, y_train)

# 3. Predict Probabilities
# Returns [Prob_Loss, Prob_Win]
probs = model.predict_proba(X_test)
p_win = probs[:, 1]
""", language="python")

st.markdown("---")

# 6. Interactive Viz
st.header("5. Interactive Visualization")
st.markdown("Train a Logistic Regression model on a toy dataset and see the decision boundary.")

col1, col2 = st.columns([1, 3])

with col1:
    dataset_type = st.selectbox("Dataset", ["Linear", "Moons"])
    noise = st.slider("Noise", 0.0, 1.0, 0.2)
    C_param = st.slider("Regularization (C)", 0.01, 10.0, 1.0, help="Lower C = Stronger Regularization (Simpler boundary)")

with col2:
    # Generate Data
    if dataset_type == "Linear":
        X, y = generate_linear(noise=noise)
    else:
        X, y = generate_moons(noise=noise)

    # Train Model
    clf = LogisticRegression(C=C_param)
    clf.fit(X, y)

    # Plot Decision Boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # Contour (Probability)
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.02),
        y=np.arange(y_min, y_max, 0.02),
        z=Z,
        colorscale='RdBu',
        opacity=0.8,
        showscale=False
    ))

    # Scatter (Data points)
    fig.add_trace(go.Scatter(
        x=X[y==0, 0], y=X[y==0, 1], mode='markers', name='Class 0',
        marker=dict(color='red', line=dict(width=1, color='black'))
    ))
    fig.add_trace(go.Scatter(
        x=X[y==1, 0], y=X[y==1, 1], mode='markers', name='Class 1',
        marker=dict(color='blue', line=dict(width=1, color='black'))
    ))

    fig.update_layout(title=f"Logistic Regression Decision Boundary (Acc: {clf.score(X, y):.2f})", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Notice how Logistic Regression can only form a straight line (linear boundary). It fails on the 'Moons' dataset unless we add polynomial features!")

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Try Logistic Regression in the Playground", icon="🎮")
