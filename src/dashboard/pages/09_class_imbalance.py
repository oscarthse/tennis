import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.model_cards import render_model_card

st.set_page_config(page_title="Class Imbalance", page_icon="⚖️", layout="wide")
sidebar_navigation()

st.title("⚖️ Class Imbalance: The Silent Killer")

# 1. Intuition
st.header("1. Intuition: The Classroom Analogy")
st.markdown("""
### The Problem
Imagine a classroom with **100 people**:
*   **95 Students** (Majority Class)
*   **5 Teachers** (Minority Class)

Your goal is to build a model that identifies the Teachers.

If you use a standard model, it will quickly learn a simple cheat: **"Just guess 'Student' for everyone."**
*   Accuracy: 95% (A+ grade!)
*   Teachers Found: 0% (F grade!)

This happens because the model is rewarded for being right *on average*, and the students dominate the average.

### The Solution: Weighted Loss
To fix this, we need to change the rules of the game. We tell the model:
*   "If you mistake a Student, you lose **1 point**."
*   "If you mistake a Teacher, you lose **20 points**."

Now, the model cannot afford to ignore the Teachers. It will try much harder to find them, even if it means making a few mistakes on the Students.
""")

st.markdown("---")

# 2. Math
st.header("2. The Math: Weighted Loss Functions")
st.markdown("Standard Loss functions treat every data point equally. Weighted Loss assigns a weight $w_c$ to each class $c$.")

st.latex(r"J(\theta) = - \sum_{i=1}^N w_{y_i} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]")

st.markdown("""
*   $w_{y_i}$: The weight for the class of sample $i$.
*   **Majority Class (0)**: Weight $w_0 = 1$.
*   **Minority Class (1)**: Weight $w_1 = \frac{N_{total}}{2 \cdot N_{minority}}$ (Inverse Frequency).
""")

st.subheader("Example Calculation")
st.markdown("Suppose we have 100 samples: 95 Class 0, 5 Class 1.")
st.latex(r"w_0 \approx 1")
st.latex(r"w_1 = \frac{100}{2 \cdot 5} = 10")
st.markdown("The model is penalized **10 times more** for missing a Class 1 sample.")

st.markdown("---")

# 3. Interactive Viz
st.header("3. Interactive Visualization")
st.markdown("See how `class_weight='balanced'` shifts the decision boundary to save the minority class.")

col1, col2 = st.columns([1, 3])

with col1:
    imbalance_ratio = st.slider("Minority Class Ratio", 0.01, 0.5, 0.05)
    use_weights = st.checkbox("Use Class Weights?", value=False)

    st.info(f"Class 0: {1-imbalance_ratio:.0%} | Class 1: {imbalance_ratio:.0%}")
    if use_weights:
        w1 = 1 / (2 * imbalance_ratio)
        st.success(f"Weight for Class 1: {w1:.1f}x")

with col2:
    # Generate Imbalanced Data
    n_samples = 500
    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority

    X, y = make_classification(
        n_samples=n_samples,
        n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1,
        weights=[1-imbalance_ratio, imbalance_ratio],
        class_sep=0.5, random_state=42
    )

    # Train Model
    weights = 'balanced' if use_weights else None
    clf = SVC(kernel='linear', class_weight=weights, C=1.0)
    clf.fit(X, y)

    # Plot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z, colorscale='RdBu', opacity=0.4, showscale=False))
    fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', name='Majority (0)', marker=dict(color='red', opacity=0.5)))
    fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', name='Minority (1)', marker=dict(color='blue', size=10, line=dict(width=2, color='white'))))

    title = "Balanced SVM" if use_weights else "Standard SVM"
    recall_1 = np.mean(clf.predict(X[y==1]) == 1)
    fig.update_layout(title=f"{title} (Recall Class 1: {recall_1:.2%})", height=500)
    st.plotly_chart(fig, use_container_width=True)

    if not use_weights and imbalance_ratio < 0.1:
        st.warning("⚠️ Notice how the Standard SVM ignores the blue points? It just predicts 'Red' for almost everything!")
    if use_weights:
        st.success("✅ With Class Weights, the boundary moves to capture the blue points, even though they are rare.")

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Back to Playground", icon="🎮")
