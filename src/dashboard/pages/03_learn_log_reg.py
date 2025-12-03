import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.mermaid import render_mermaid
from src.dashboard.components.toy_datasets import generate_moons, generate_linear

st.set_page_config(page_title="Logistic Regression", page_icon="📈", layout="wide")
sidebar_navigation()

st.title("📈 Logistic Regression: The Foundation")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown(r"""
Logistic Regression is a **Probabilistic Linear Classifier**. It predicts the probability $P(y=1|x)$ that an instance belongs to the positive class.

**The Model Equation:**
""")
st.latex(r"P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}")
st.markdown(r"""
*   $x$: Input feature vector.
*   $w$: Weight vector (Feature importance).
*   $b$: Bias term (Intercept).
*   $\sigma(z)$: The **Sigmoid Function**. It squashes any number $z \in (-\infty, \infty)$ into the range $(0, 1)$.
""")

# --- 2. Geometry / Structure ---
st.header("2. Geometry: The Linear Boundary")
st.markdown(r"""
Although the output is a curve (S-shape), the **Decision Boundary** is a straight line (or hyperplane).

**Why?**
We predict Class 1 if $P > 0.5$.
This happens exactly when the input to the sigmoid is positive:
""")
st.latex(r"w^T x + b > 0")
st.markdown(r"""
This equation $w^T x + b = 0$ defines a **Hyperplane** that cuts the space in two.
*   **One side**: Probability $> 0.5$ (Win).
*   **Other side**: Probability $< 0.5$ (Lose).
*   **On the line**: Probability $= 0.5$ (Uncertain).
""")

render_mermaid("""
graph LR
    Input["Inputs (x)"] -->|Dot Product| Linear["Linear Score (z)"]
    Linear -->|Sigmoid| Prob["Probability (p)"]
    Prob -->|Threshold| Pred["Prediction (y)"]

    style Linear fill:#fff3e0
    style Prob fill:#e8f5e9
""", height=200)

# --- 3. The Mathematical "Why": Odds & Log-Odds ---
st.header("3. The Mathematical Intuition: From Odds to Sigmoid")
st.markdown(r"""
Why do we use the **Sigmoid** function? It's not arbitrary. It comes from a very natural assumption about **Log-Odds**.

### 3.1. The Problem with Linear Regression
If we tried to use standard Linear Regression ($y = w^T x + b$) for classification:
1.  **Range Issue**: Linear regression predicts values from $-\infty$ to $+\infty$. Probabilities must be in $[0, 1]$.
2.  **Meaning**: What does a prediction of 150 mean? Or -20?

We need a link function to bridge the gap between the linear world ($-\infty, \infty$) and the probability world ($0, 1$).

### 3.2. Step 1: Probability to Odds
First, let's talk about **Odds**. If the probability of winning is $P$, the odds are:
""")
st.latex(r"\text{Odds} = \frac{P}{1-P}")
st.markdown(r"""
*   **Range**: If $P \in [0, 1)$, then $\text{Odds} \in [0, \infty)$.
*   **Example**: If $P=0.8$ (80% chance), Odds = $0.8 / 0.2 = 4$. We say "4 to 1 odds".

### 3.3. Step 2: Odds to Log-Odds (Logit)
The range $[0, \infty)$ is better, but still restricted (must be positive).
Let's take the **Natural Logarithm** of the odds. This is called the **Logit** function.
""")
st.latex(r"\text{Log-Odds} = \ln(\text{Odds}) = \ln\left(\frac{P}{1-P}\right)")
st.markdown(r"""
*   **Range**: If Odds $\in [0, \infty)$, then Log-Odds $\in (-\infty, \infty)$.
*   **Symmetry**:
    *   $P=0.5 \implies \text{Odds}=1 \implies \text{Log-Odds}=0$.
    *   $P=0.9 \implies \text{Log-Odds} \approx 2.2$.
    *   $P=0.1 \implies \text{Log-Odds} \approx -2.2$.

### 3.4. Step 3: The Linear Assumption
Now we have a quantity, **Log-Odds**, that spans $(-\infty, \infty)$, just like a linear equation!
So, **Logistic Regression makes one simple assumption**:
> **The Log-Odds are Linear with respect to the input features.**
""")
st.latex(r"\ln\left(\frac{P}{1-P}\right) = w^T x + b")
st.markdown(r"""
This is the heart of the model. We are modeling the *log-odds* linearly.

### 3.5. Step 4: Solving for P (The Derivation)
Now, let's solve for $P$ to get our prediction function.
Let $z = w^T x + b$.
""")
st.latex(r"""
\begin{aligned}
\ln\left(\frac{P}{1-P}\right) &= z \\
\frac{P}{1-P} &= e^z \quad \text{(Exponentiate both sides)} \\
P &= e^z (1 - P) \\
P &= e^z - P \cdot e^z \\
P + P \cdot e^z &= e^z \\
P(1 + e^z) &= e^z \\
P &= \frac{e^z}{1 + e^z} \\
P &= \frac{1}{1 + e^{-z}} \quad \text{(Divide top and bottom by } e^z \text{)}
\end{aligned}
""")
st.markdown("""
**Voila!** We have derived the **Sigmoid Function**.
This explains *why* the sigmoid function is used. It's the natural consequence of assuming the log-odds are linear.
""")

# --- 4. Constraints / Objective / Loss ---
st.header("4. The Optimization Problem")
st.markdown(r"""
We cannot use "Least Squares" (MSE) because it assumes errors are Gaussian (they are not; they are binary).
We use **Maximum Likelihood Estimation (MLE)**.

**The Goal:**
Find $w, b$ that maximize the probability of the observed data.
""")
st.latex(r"\text{Likelihood} = \prod_{i=1}^N P(y_i | x_i; w)")

st.markdown(r"""
**The Loss Function (Log Loss / Cross-Entropy):**
To make it easier to optimize, we take the Negative Logarithm.
""")
st.latex(r"J(w) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]")
st.markdown(r"""
*   **If $y=1$**: We want $\hat{y} \approx 1$. If $\hat{y} \approx 0$, $\log(\hat{y}) \to -\infty$, so Loss $\to \infty$.
*   **If $y=0$**: We want $\hat{y} \approx 0$. If $\hat{y} \approx 1$, $\log(1-\hat{y}) \to -\infty$, so Loss $\to \infty$.
""")

# --- 5. Deeper Components (Gradient) ---
st.header("5. The Gradient Update")
st.markdown(r"""
How do we find the best $w$? We use **Gradient Descent**.
The derivative of the Log Loss with respect to weights is surprisingly simple:
""")
st.latex(r"\nabla_w J = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) x_i")
st.markdown(r"""
*   **Error**: $(\hat{y} - y)$. The difference between prediction and reality.
*   **Input**: $x$. The direction of the feature.
*   **Update Rule**: $w \leftarrow w - \eta \cdot \text{Error} \cdot x$.
*   **Interpretation**: If we predicted too high ($\hat{y} > y$), we push $w$ in the opposite direction of $x$.
""")

# --- 6. What the Solution Looks Like ---
st.header("6. The Solution: Interpreting Weights")
st.markdown(r"""
What does the model actually learn? It learns the **Log-Odds** of the positive class.
""")
st.latex(r"\ln \left( \frac{P(y=1)}{P(y=0)} \right) = w^T x + b")
st.markdown(r"""
*   If $w_1 = 2.0$: Increasing $x_1$ by 1 unit increases the **Log-Odds** of winning by 2.0.
*   This means the **Odds** increase by factor $e^2 \approx 7.4$.
*   This makes Logistic Regression highly **Interpretable**.
""")

# --- 7. Visualization ---
st.header("7. Visualization")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    C_param = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
    dataset = st.selectbox("Dataset", ["Linear", "Moons"])
    noise = st.slider("Noise", 0.0, 1.0, 0.2)

with col_viz:
    if dataset == "Linear":
        X, y = generate_linear(n_samples=200, noise=noise)
    else:
        X, y = generate_moons(n_samples=200, noise=noise)

    clf = LogisticRegression(C=C_param)
    clf.fit(X, y)

    # Grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig = go.Figure()
    # Probability Contour
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z,
                             colorscale='RdBu', showscale=True,
                             contours=dict(start=0, end=1, size=0.1, coloring='heatmap')))

    # Decision Boundary Line (p=0.5)
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z,
                             colorscale='Greys', showscale=False,
                             contours=dict(start=0.5, end=0.5, size=0.1, coloring='lines', showlabels=True)))

    # Data
    fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', name='Class 0', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', name='Class 1', marker=dict(color='blue')))

    fig.update_layout(title=f"Logistic Regression (C={C_param})", height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- 8. How to do it in Python ---
st.header("8. How to do it in Python 🐍")
st.code("""
from sklearn.linear_model import LogisticRegression

# 1. Initialize
model = LogisticRegression(C=1.0, solver='lbfgs')

# 2. Train
model.fit(X_train, y_train)

# 3. Predict Class (0 or 1)
y_pred = model.predict(X_test)

# 4. Predict Probability (0.0 to 1.0)
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Inspect Weights
print(f"Weights: {model.coef_}")
print(f"Bias: {model.intercept_}")
""", language="python")

# --- 9. Hyperparameters & Behavior ---
st.header("9. Hyperparameters & Behavior")
st.markdown(r"""
*   **C (Inverse Regularization)**:
    *   **High C**: Weak regularization. Model trusts training data more. Can overfit.
    *   **Low C**: Strong regularization. Pushes weights $w$ towards zero. Simpler model.
*   **Penalty**: L1 (Lasso) or L2 (Ridge). L1 can make weights exactly zero (Feature Selection).
""")

# --- 10. Super Summary ---
st.header("10. Super Summary 🦸")
st.info(r"""
*   **Goal**: Estimate probability of class 1.
*   **Math**: Minimize Log Loss (Cross-Entropy).
*   **Key Insight**: It's a linear model wrapped in a sigmoid.
*   **Knobs**: $C$ controls complexity.
""")
