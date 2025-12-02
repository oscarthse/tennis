import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.model_cards import render_model_card
from src.dashboard.components.toy_datasets import generate_moons, generate_circles, generate_linear

st.set_page_config(page_title="SVM & KNN", page_icon="📐", layout="wide")
sidebar_navigation()

st.title("📐 SVM & K-Nearest Neighbors")

tab1, tab2 = st.tabs(["K-Nearest Neighbors (KNN)", "Support Vector Machines (SVM)"])

with tab1:
    # 1. TL;DR
    render_model_card(
        title="K-Nearest Neighbors",
        description="A simple, non-parametric method that predicts the class of a point based on the majority class of its 'k' nearest neighbors.",
        pros=["Simple to understand", "No training phase (lazy learning)", "Adapts to local structure"],
        cons=["Slow prediction (must calculate distance to all points)", "Sensitive to scale (needs normalization)"]
    )

    st.markdown("---")

    # 2. Intuition
    st.header("1. Intuition: 'Tell me who your friends are...'")
    st.markdown("""
    ### The Tennis Analogy
    Suppose we have a new match: **Alcaraz vs. Sinner**. We want to predict the winner.

    KNN says:
    1.  Look at history. Find the **5 matches** that looked *most similar* to this one (similar ranks, similar surface, similar odds).
    2.  Who won those matches?
        *   Match 1: Favorite Won.
        *   Match 2: Favorite Won.
        *   Match 3: Underdog Won.
        *   Match 4: Favorite Won.
        *   Match 5: Favorite Won.
    3.  **Vote**: 4 Favorites, 1 Underdog.
    4.  **Prediction**: Favorite Wins.

    **Key Concept**: "Similarity". We measure similarity using **Distance**.
    """)

    # 3. Math
    st.header("2. The Math: Euclidean Distance")
    st.markdown("How do we calculate the distance between two matches $p$ and $q$? We treat them as points in space.")
    st.latex(r"d(p, q) = \sqrt{\sum_{i=1}^n (q_i - p_i)^2}")

    st.markdown("""
    *   $q_i, p_i$: The values of feature $i$ (e.g., Rank Diff, Points Diff).
    *   We sum the squared differences and take the square root.
    """)

    st.warning("⚠️ **Crucial Step: Scaling**. If 'Points' ranges from 0-10000 and 'Rank' ranges from 1-100, 'Points' will dominate the distance. We MUST scale features (e.g., to 0-1) so they contribute equally.")

    # 4. Worked Example
    st.header("3. Worked Example")
    st.markdown("Let's find the distance between Match A and Match B.")

    st.markdown("""
    *   **Match A**: RankDiff = 10, Odds = 1.5
    *   **Match B**: RankDiff = 12, Odds = 1.6
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Step 1: Differences")
        st.latex(r"\Delta \text{Rank} = 12 - 10 = 2")
        st.latex(r"\Delta \text{Odds} = 1.6 - 1.5 = 0.1")

    with col2:
        st.markdown("#### Step 2: Euclidean Distance")
        st.latex(r"d = \sqrt{2^2 + 0.1^2}")
        st.latex(r"d = \sqrt{4 + 0.01} = \sqrt{4.01} \approx 2.002")

    st.markdown("The distance is dominated by Rank (2.0) vs Odds (0.1). This shows why scaling is needed!")

    st.markdown("---")

    # 5. Interactive Viz
    st.header("4. Interactive Visualization")

    col1, col2 = st.columns([1, 3])
    with col1:
        k_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 3)
        dataset_knn = st.selectbox("Dataset (KNN)", ["Moons", "Circles", "Linear"])

    with col2:
        if dataset_knn == "Moons":
            X, y = generate_moons()
        elif dataset_knn == "Circles":
            X, y = generate_circles()
        else:
            X, y = generate_linear()

        clf = KNeighborsClassifier(n_neighbors=k_neighbors)
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
        fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', marker=dict(color='red', line=dict(width=1, color='black'))))
        fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', marker=dict(color='blue', line=dict(width=1, color='black'))))

        fig.update_layout(title=f"KNN (k={k_neighbors}) Boundary", height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # 1. TL;DR
    render_model_card(
        title="Support Vector Machine",
        description="Finds the hyperplane (line in 2D) that separates classes with the **maximum margin** (widest gap). It tries to build the widest possible road between the two classes.",
        pros=["Effective in high dimensions", "Robust to overfitting (with C)", "Versatile (kernels)"],
        cons=["Hard to interpret probability", "Sensitive to noise", "Slow on large datasets"]
    )

    st.markdown("---")

    # 2. Intuition
    st.header("1. Intuition: The Widest Road")
    st.markdown("""
    Imagine you have red balls (Losses) and blue balls (Wins) on a table. You want to place a stick (line) to separate them.

    *   **Option A**: Place the stick close to the red balls. (Risky).
    *   **Option B**: Place the stick close to the blue balls. (Risky).
    *   **Option C**: Place the stick exactly in the middle, maximizing the gap to both. (**Best**).

    SVM finds Option C. The "gap" is called the **Margin**. The points touching the margin are the **Support Vectors** (the most difficult matches to classify).
    """)

    # 3. Math
    st.header("2. The Math: Hinge Loss")
    st.markdown("We want to minimize the error AND maximize the margin. This leads to the **Hinge Loss** objective:")

    st.latex(r"J(w) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^N \max(0, 1 - y_i(w^\top x_i + b))")

    st.markdown("### Why minimize $||w||^2$?")
    st.markdown("This is not arbitrary. It is a **Geometric Proof**.")
    st.markdown("1. The distance from a point $x$ to the hyperplane $w^Tx + b = 0$ is $\\frac{|w^Tx + b|}{||w||}$.")
    st.markdown("2. For Support Vectors, $|w^Tx + b| = 1$.")
    st.markdown("3. So the distance (Margin) is $\\frac{1}{||w||}$.")
    st.markdown("4. To **Maximize** the Margin $\\frac{1}{||w||}$, we must **Minimize** $||w||$.")
    st.markdown("5. For mathematical convenience (derivatives), we minimize $\\frac{1}{2}||w||^2$.")

    st.markdown("""
    *   **Term 1**: $\frac{1}{2} ||w||^2$. Minimizing $w$ maximizes the Margin. (Geometric property).
    *   **Term 2**: The Error.
        *   If point is correctly classified and outside the margin ($y \cdot f(x) \ge 1$), Loss is 0.
        *   If point is inside the margin or wrong side, Loss increases linearly.
    *   **C (Regularization)**: Controls the trade-off.
        *   **High C**: Care more about not making mistakes (Hard Margin).
        *   **Low C**: Care more about a wide margin, allowing some mistakes (Soft Margin).
    """)

    st.markdown("---")

    # 4. Interactive Viz
    st.header("3. Interactive Visualization")

    col1, col2 = st.columns([1, 3])
    with col1:
        C_svm = st.slider("Regularization (C)", 0.01, 10.0, 1.0, help="High C = Strict (Complex), Low C = Loose (Simple)")
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"], help="RBF allows curved boundaries.")
        dataset_svm = st.selectbox("Dataset (SVM)", ["Linear", "Moons", "Circles"])

    with col2:
        if dataset_svm == "Moons":
            X, y = generate_moons()
        elif dataset_svm == "Circles":
            X, y = generate_circles()
        else:
            X, y = generate_linear()

        clf = SVC(C=C_svm, kernel=kernel)
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
        fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', marker=dict(color='red', line=dict(width=1, color='black'))))
        fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', marker=dict(color='blue', line=dict(width=1, color='black'))))

        fig.update_layout(title=f"SVM ({kernel}, C={C_svm}) Boundary", height=500)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Try SVM/KNN in the Playground", icon="🎮")
