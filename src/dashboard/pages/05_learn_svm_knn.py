import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.model_cards import render_model_card
from src.dashboard.components.toy_datasets import generate_moons, generate_circles, generate_linear
from src.dashboard.components.mermaid import render_mermaid

st.set_page_config(page_title="SVM & KNN", page_icon="📐", layout="wide")
sidebar_navigation()

st.title("📐 SVM & K-Nearest Neighbors")

tab1, tab2 = st.tabs(["K-Nearest Neighbors (KNN)", "Support Vector Machines (SVM)"])

# ==========================================
# KNN SECTION
# ==========================================
with tab1:
    # --- LAYER 1: Intuition ---
    st.header("1. Intuition: 'Tell me who your friends are...' 🤝")
    st.markdown("""
    If you walk into a room and everyone is speaking Italian, you are probably in Italy.
    KNN uses this simple logic: **You are what your neighbors are.**

    To classify a new tennis match:
    1.  Find the 5 most similar matches from the past.
    2.  See who won those matches.
    3.  Vote.
    """)
    st.markdown("---")

    # --- LAYER 2: Analogy ---
    st.header("2. Analogy: Real Estate Pricing 🏠")
    st.markdown("""
    How do you price a house? You look at "Comps" (Comparables).
    *   You find 3 houses nearby with similar size.
    *   They sold for \$300k, \$310k, and \$290k.
    *   You average them to price your house at \$300k.

    KNN is just "Comps" for classification.
    """)
    st.markdown("---")

    # --- LAYER 3: Structure ---
    st.header("3. Structure: The Distance Metric 📏")
    st.markdown("The core of KNN is measuring 'Similarity'. We do this with **Distance**.")
    st.latex(r"Distance(A, B) = \sqrt{(x_A - x_B)^2 + (y_A - y_B)^2}")
    st.markdown("This is **Euclidean Distance** (straight line).")
    st.markdown("---")

    # --- LAYER 4: Step-by-Step ---
    st.header("4. Step-by-Step Prediction 👣")
    st.markdown("**New Match**: RankDiff=10, Odds=1.5")
    st.markdown("**History**:")
    st.markdown("1. Match A (RankDiff=9, Odds=1.4) -> **Win**")
    st.markdown("2. Match B (RankDiff=11, Odds=1.6) -> **Win**")
    st.markdown("3. Match C (RankDiff=50, Odds=5.0) -> **Lose**")

    st.markdown("**Step 1: Calculate Distances**")
    st.latex(r"d(New, A) = \sqrt{(10-9)^2 + (1.5-1.4)^2} \approx 1.0")
    st.latex(r"d(New, B) = \sqrt{(10-11)^2 + (1.5-1.6)^2} \approx 1.0")
    st.latex(r"d(New, C) = \sqrt{(10-50)^2 + (1.5-5.0)^2} \approx 40.1")

    st.markdown("**Step 2: Find Neighbors (k=3)**")
    st.markdown("The closest are Match A and Match B. (Match C is too far).")

    st.markdown("**Step 3: Vote**")
    st.markdown("A (Win) + B (Win) = **2 Wins**. Prediction: **Win**.")
    st.markdown("---")

    # --- LAYER 5: Math ---
    st.header("5. The Math: Minkowski Distance 🧮")
    st.markdown("Euclidean is just one type. The general form is **Minkowski Distance**:")
    st.latex(r"D(x, y) = \left( \sum_{i=1}^n |x_i - y_i|^p \right)^{1/p}")
    st.markdown("""
    *   $p=1$: **Manhattan Distance** (Taxicab geometry).
    *   $p=2$: **Euclidean Distance** (Straight line).
    """)
    st.markdown("---")

    # --- LAYER 9: Exercises ---
    st.header("9. Exercises 📝")
    st.info("Calculate the Euclidean distance between point (0,0) and (3,4). (Hint: 3-4-5 Triangle)")


# ==========================================
# SVM SECTION
# ==========================================
with tab2:
    # --- LAYER 1: Intuition ---
    st.header("1. Intuition: The Widest Road 🛣️")
    st.markdown("""
    Imagine you are building a road to separate two cities (Red City and Blue City).
    *   You want the road to be **as wide as possible** for safety.
    *   The "Margin" is the width of the road.
    *   The "Support Vectors" are the buildings right on the edge of the road.
    """)
    st.markdown("---")

    # --- LAYER 3: Structure ---
    st.header("3. Structure: Hyperplanes and Margins")
    render_mermaid("""
    graph TD
        N1["Data Points"] --> N2["Hyperplane - Line"]
        N2 --> N3["Margin - Gap"]
        N3 --> N4["Maximize Width"]
    """, height=200)
    st.markdown("---")

    # --- LAYER 5: Full Math ---
    st.header("5. The Math: Hinge Loss & Geometry 🧮")

    st.subheader("A. The Geometric Margin")
    st.markdown("The distance from the hyperplane is $\\frac{1}{||w||}$.")
    st.markdown("To **Maximize Distance**, we must **Minimize $||w||$**.")

    st.subheader("B. The Objective Function")
    st.latex(r"J(w) = \frac{1}{2}||w||^2 + C \sum_{i=1}^N \max(0, 1 - y_i(w^T x_i + b))")
    st.markdown("""
    *   **Part 1**: Make the road wide ($||w||^2$).
    *   **Part 2**: Don't crash into buildings (Loss).
    *   **C**: The trade-off. High C = "Don't crash" (Strict). Low C = "Wide road" (Loose).
    """)

    st.subheader("C. The Kernel Trick")
    st.markdown("What if a straight road isn't enough? We warp space!")
    st.latex(r"K(x, y) = \exp(-\gamma ||x - y||^2)")
    st.markdown("This (RBF Kernel) measures similarity. It lifts data into infinite dimensions where it becomes separable.")
    st.markdown("---")

    # --- Interactive Viz ---
    st.header("10. Interactive Playground")
    col1, col2 = st.columns([1, 3])
    with col1:
        C_svm = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        dataset_svm = st.selectbox("Dataset", ["Moons", "Circles"])

    with col2:
        if dataset_svm == "Moons":
            X, y = generate_moons()
        else:
            X, y = generate_circles()

        clf = SVC(C=C_svm, kernel=kernel)
        clf.fit(X, y)

        # Plot
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = go.Figure()
        fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z, colorscale='RdBu', opacity=0.4, showscale=False))
        fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', marker=dict(color='red')))
        fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', marker=dict(color='blue')))
        fig.update_layout(title=f"SVM ({kernel}) Boundary", height=500)
        st.plotly_chart(fig, use_container_width=True)

st.page_link("pages/02_model_playground.py", label="🎮 Go to Playground", icon="🎮")
