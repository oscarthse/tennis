import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.toy_datasets import generate_moons

st.set_page_config(page_title="Class Imbalance", page_icon="⚖", layout="wide")
sidebar_navigation()

st.title("⚖ Class Imbalance")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown("""
In many problems (Fraud, Disease, Betting), one class is much rarer than the other.
*   **Majority Class**: The common one (e.g., "Not Fraud").
*   **Minority Class**: The rare one (e.g., "Fraud").

If the ratio is 99:1, a model can achieve **99% Accuracy** by just saying "Not Fraud" every time. This is useless.
""")

# --- 2. Geometry / Structure ---
st.header("2. Geometry: The Swamped Boundary")
st.markdown("""
Standard algorithms minimize **Total Error**.
Since the Majority Class contributes 99% of the error terms, the model focuses entirely on getting them right.
The Decision Boundary is pushed deep into the Minority territory, ignoring the rare points.
""")

# --- 3. Constraints / Objective / Loss ---
st.header("3. The Fix: Weighted Loss")
st.markdown("""
We can tell the model: "Pay more attention to the rare guys."
We introduce **Class Weights** ($w_0, w_1$).

**Weighted Log Loss:**
""")
st.latex(r"J(w) = - \frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \log(\hat{y}_i) + w_0 (1 - y_i) \log(1 - \hat{y}_i) \right]")
st.markdown("""
*   If Class 1 is 10x rarer, we set $w_1 = 10$ and $w_0 = 1$.
*   Now, making a mistake on a Class 1 example costs 10x more penalty.
*   The model is forced to respect the minority class.
""")

# --- 6. Visualization ---
st.header("6. Visualization: The Shift")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    weight = st.slider("Minority Weight", 1, 20, 1)

with col_viz:
    # Imbalanced Data
    X, y = generate_moons(n_samples=300, noise=0.3)
    # Kill 90% of class 1
    mask = (y == 0) | (np.random.rand(len(y)) < 0.1)
    X = X[mask]
    y = y[mask]

    clf = SVC(kernel='linear', class_weight={1: weight, 0: 1}, probability=True)
    clf.fit(X, y)

    # Grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z,
                             colorscale='RdBu', showscale=False, contours=dict(start=0, end=0, size=1, coloring='lines')))
    fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', name='Majority (0)', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', name='Minority (1)', marker=dict(color='blue', size=10)))

    fig.update_layout(title=f"Decision Boundary (Weight 1:{weight})", height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- 8. Super Summary ---
st.header("8. Super Summary 🦸")
st.info("""
*   **Goal**: Prevent the majority class from dominating.
*   **Math**: Weighted Loss Function ($w_1 > w_0$).
*   **Key Insight**: Making mistakes on rare items must be expensive.
*   **Knobs**: Class Weights or Resampling Ratios.
""")
