import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.mermaid import render_mermaid
from src.dashboard.components.toy_datasets import generate_moons, generate_circles


st.set_page_config(page_title="Trees & Forests", page_icon="🌳", layout="wide")
sidebar_navigation()

st.title("🌳 Decision Trees & Random Forests")

tab1, tab2 = st.tabs(["Decision Trees", "Random Forests"])

# ==========================================
# DECISION TREES
# ==========================================
with tab1:
    st.header("Decision Trees")

    # --- 1. Core Model Definition ---
    st.subheader("1. Core Model Definition")
    st.markdown("""
    A Decision Tree is a **Non-Parametric** model that learns a hierarchy of "If-Then" rules.
    It performs **Recursive Partitioning**: splitting the data into smaller, purer subsets.

    **The Prediction Function:**
    """)
    st.latex(r"\hat{y}(x) = \sum_{m=1}^M c_m I(x \in R_m)")
    st.markdown(r"""
    *   $M$: Number of leaf nodes (regions).
    *   $R_m$: The $m$-th region (box) in the feature space.
    *   $c_m$: The constant prediction for region $R_m$ (e.g., majority class).
    *   $I(\cdot)$: Indicator function (1 if $x$ is in $R_m$, else 0).
    """)

    # --- 2. Geometry / Structure ---
    st.subheader("2. Geometry: Orthogonal Boundaries")
    st.markdown(r"""
    Trees cut the space using **Axis-Aligned Splits** ($x_j \le t$).
    *   This creates "Boxy" decision boundaries.
    *   Unlike SVM or Logistic Regression, Trees cannot draw diagonal lines easily (they need a "staircase" to approximate a diagonal).
    """)

    render_mermaid("""
    graph TD
        Root["Root: Is RankDiff < 0?"] -->|Yes| Left["Node A: Is PointsDiff > 500?"]
        Root -->|No| Right["Node B: Is Surface = Clay?"]
        Left -->|Yes| L1["Leaf: WIN"]
        Left -->|No| L2["Leaf: LOSE"]
        Right -->|Yes| L3["Leaf: WIN"]
        Right -->|No| L4["Leaf: LOSE"]
    """, height=250)

    # --- 3. Constraints / Objective / Loss ---
    st.subheader("3. The Optimization Problem")
    st.markdown("""
    We want to find the split $(j, t)$ that maximizes the **purity** of the child nodes.
    We use a "Greedy" approach (CART algorithm).

    **The Objective (Maximize Information Gain):**
    """)
    st.latex(r"\max_{j, t} \left[ I(D_p) - \left( \frac{N_{left}}{N_p} I(D_{left}) + \frac{N_{right}}{N_p} I(D_{right}) \right) \right]")
    st.markdown("""
    *   $I(D)$: Impurity of a dataset node.
    *   $N$: Number of samples.
    *   **Goal**: Make the weighted average impurity of children much lower than the parent.
    """)

    # --- 4. Deeper Components (Impurity Metrics) ---
    st.subheader("4. Impurity Metrics: Gini vs Entropy")
    st.markdown("How do we measure 'Messiness'?")

    col_gini, col_ent = st.columns(2)
    with col_gini:
        st.markdown("**A. Gini Impurity (Default)**")
        st.latex(r"Gini = 1 - \sum_{k=1}^K p_k^2")
        st.markdown("""
        *   Measures probability of misclassification.
        *   Range: [0, 0.5] (for binary).
        *   Faster to compute (no logs).
        """)
    with col_ent:
        st.markdown("**B. Entropy (Information Theory)**")
        st.latex(r"Entropy = - \sum_{k=1}^K p_k \log_2(p_k)")
        st.markdown("""
        *   Measures "Surprise" or disorder.
        *   Range: [0, 1.0] (for binary).
        *   Tends to produce slightly more balanced trees.
        """)

    # --- 6. Visualization ---
    st.subheader("6. Visualization")

    col_viz, col_controls = st.columns([3, 1])
    with col_controls:
        depth = st.slider("Max Depth", 1, 10, 3)
        criterion = st.selectbox("Criterion", ["gini", "entropy"])
        dataset = st.selectbox("Dataset", ["Moons", "Circles"], key="tree_data")

    with col_viz:
        if dataset == "Moons":
            X, y = generate_moons(n_samples=200, noise=0.2)
        else:
            X, y = generate_circles(n_samples=200, noise=0.1)

        clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion)
        clf.fit(X, y)

        # Grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = go.Figure()
        fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02), z=Z,
                                 colorscale='RdBu', opacity=0.4, showscale=False))
        fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', name='Class 0', marker=dict(color='red')))
        fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', name='Class 1', marker=dict(color='blue')))

        fig.update_layout(title=f"Decision Tree (Depth={depth})", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# RANDOM FORESTS
# ==========================================
with tab2:
    st.header("Random Forests")

    # --- 1. Core Model Definition ---
    st.subheader("1. Core Model Definition")
    st.markdown("""
    A Random Forest is an **Ensemble** of Decision Trees. It uses **Bagging** (Bootstrap Aggregating) to reduce variance.

    **The Prediction:**
    """)
    st.latex(r"\hat{y}_{RF}(x) = \text{mode} \{ \hat{y}_1(x), \hat{y}_2(x), \dots, \hat{y}_B(x) \}")
    st.markdown("It takes a **Majority Vote** of $B$ trees.")

    # --- 3. Optimization (Variance Reduction) ---
    st.subheader("3. Why it Works: Variance Reduction")
    st.markdown("""
    A single tree is **High Variance** (unstable). If you change one data point, the whole tree might change.
    A Forest averages out these errors.

    **Variance of the Average:**
    """)
    st.latex(r"Var(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2")
    st.markdown(r"""
    *   $\sigma^2$: Variance of a single tree.
    *   $B$: Number of trees.
    *   $\rho$: Correlation between trees.
    *   **Goal**: Reduce $\rho$ (make trees diverse) and increase $B$.
    """)

    # --- 4. Deeper Components (Randomness) ---
    st.subheader("4. Injecting Randomness")
    st.markdown(r"""
    To make trees diverse (low $\rho$), we inject randomness in two places:
    1.  **Bootstrapping**: Each tree sees a random subset of the data (with replacement).
    2.  **Feature Randomness**: At each split, the tree can only choose from a random subset of features (e.g., $\sqrt{p}$ features).
    """)

    # --- 6. Visualization ---
    st.subheader("6. Visualization")

    col_viz_rf, col_controls_rf = st.columns([3, 1])
    with col_controls_rf:
        n_estimators = st.slider("Num Trees", 1, 50, 10)
        dataset_rf = st.selectbox("Dataset", ["Moons", "Circles"], key="rf_data")

    with col_viz_rf:
        if dataset_rf == "Moons":
            X_rf, y_rf = generate_moons(n_samples=200, noise=0.2)
        else:
            X_rf, y_rf = generate_circles(n_samples=200, noise=0.1)

        clf_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf_rf.fit(X_rf, y_rf)

        # Grid
        x_min_rf, x_max_rf = X_rf[:, 0].min() - 0.5, X_rf[:, 0].max() + 0.5
        y_min_rf, y_max_rf = X_rf[:, 1].min() - 0.5, X_rf[:, 1].max() + 0.5
        xx_rf, yy_rf = np.meshgrid(np.arange(x_min_rf, x_max_rf, 0.02), np.arange(y_min_rf, y_max_rf, 0.02))

        Z_rf = clf_rf.predict(np.c_[xx_rf.ravel(), yy_rf.ravel()])
        Z_rf = Z_rf.reshape(xx_rf.shape)

        fig_rf = go.Figure()
        fig_rf.add_trace(go.Contour(x=np.arange(x_min_rf, x_max_rf, 0.02), y=np.arange(y_min_rf, y_max_rf, 0.02), z=Z_rf,
                                    colorscale='RdBu', opacity=0.4, showscale=False))
        fig_rf.add_trace(go.Scatter(x=X_rf[y_rf==0, 0], y=X_rf[y_rf==0, 1], mode='markers', marker=dict(color='red')))
        fig_rf.add_trace(go.Scatter(x=X_rf[y_rf==1, 0], y=X_rf[y_rf==1, 1], mode='markers', marker=dict(color='blue')))

        fig_rf.update_layout(title=f"Random Forest ({n_estimators} Trees)", height=500)
        st.plotly_chart(fig_rf, use_container_width=True)

    # --- 8. Super Summary ---
    st.subheader("8. Super Summary 🦸")
    st.info("""
    *   **Goal**: Partition space into pure boxes.
    *   **Math**: Maximize Information Gain (Gini/Entropy).
    *   **Key Insight**: Trees are "Boxy". Forests smooth out the boxes by averaging.
    *   **Knobs**: Depth (Complexity), Num Trees (Stability).
    """)
