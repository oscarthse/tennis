import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.toy_datasets import generate_moons

st.set_page_config(page_title="Class Imbalance", page_icon="⚖", layout="wide")
sidebar_navigation()

st.title("⚖ Class Imbalance: Finding Needles in Haystacks")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown(r"""
In the real world, interesting things are rare.
*   **Fraud**: 0.1% of transactions.
*   **Disease**: 1% of patients.
*   **Tennis**: Upset wins are rarer than favorite wins.

**The Accuracy Paradox**:
If you have 99% Class 0 and 1% Class 1, a model that predicts "All 0" has **99% Accuracy**.
But it has **0% Recall** for the class we care about. It is useless.
""")

# --- 2. The Solutions (Platinum Depth) ---
st.header("2. The Solutions")

tab_weight, tab_resample, tab_smote = st.tabs(["Class Weights", "Resampling", "SMOTE"])

with tab_weight:
    st.subheader("1. Class Weights (Cost-Sensitive Learning)")
    st.markdown(r"""
    We change the Loss Function.
    We tell the model: "Mistakes on the Minority Class are **Expensive**."
    """)
    st.latex(r"J(w) = - \sum [ w_{minority} \cdot y \log(\hat{y}) + w_{majority} \cdot (1-y) \log(1-\hat{y}) ]")
    st.markdown("If ratio is 1:10, set $w_{minority} = 10$. The model will try 10x harder to get them right.")

with tab_resample:
    st.subheader("2. Random Resampling")
    st.markdown(r"""
    We change the Data.
    *   **Undersampling**: Delete random Majority examples until balanced. (Risk: Loss of data).
    *   **Oversampling**: Duplicate random Minority examples until balanced. (Risk: Overfitting to duplicates).
    """)

with tab_smote:
    st.subheader("3. SMOTE (Synthetic Minority Over-sampling Technique)")
    st.markdown(r"""
    Don't just duplicate. **Create new ones.**
    1.  Pick a Minority point $A$.
    2.  Find its nearest Minority neighbor $B$.
    3.  Draw a line between $A$ and $B$.
    4.  Create a new point $C$ somewhere on that line.

    This expands the "territory" of the minority class without exact copying.
    """)

# --- 6. Visualization ---
st.header("6. Visualization: The Swamped Boundary")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    weight = st.slider("Minority Weight", 1, 50, 1)
    smote_on = st.checkbox("Simulate SMOTE")

with col_viz:
    # Imbalanced Data
    X, y = generate_moons(n_samples=400, noise=0.3)
    # Kill 95% of class 1
    mask = (y == 0) | (np.random.rand(len(y)) < 0.05)
    X = X[mask]
    y = y[mask]

    if smote_on:
        # Fake SMOTE for viz (just add jittered copies)
        X_min = X[y==1]
        new_X = []
        for _ in range(5): # 5x oversampling
            noise = np.random.normal(0, 0.1, X_min.shape)
            new_X.append(X_min + noise)
        X = np.vstack([X] + new_X)
        y = np.concatenate([y] + [np.ones(len(X_min))]*5)

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
    fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', name='Majority', marker=dict(color='red', opacity=0.3)))
    fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', name='Minority', marker=dict(color='blue', size=8)))

    fig.update_layout(title=f"Decision Boundary (Weight={weight}, SMOTE={smote_on})", height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- 8. Super Summary ---
st.header("8. Super Summary 🦸")
st.info(r"""
*   **Goal**: Don't ignore the rare event.
*   **Problem**: Standard Accuracy is misleading.
*   **Fix 1**: Use Precision/Recall/F1/AUC.
*   **Fix 2**: Use Class Weights (Penalize mistakes).
*   **Fix 3**: Use SMOTE (Synthesize data).
""")
