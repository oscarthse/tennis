import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.mermaid import render_mermaid
from src.dashboard.components.toy_datasets import generate_moons, generate_circles

st.set_page_config(page_title="Trees & Forests", page_icon="🌳", layout="wide")
sidebar_navigation()

st.title("🌳 Decision Trees & Random Forests: The Logic of Learning")

tab1, tab2 = st.tabs(["Decision Trees", "Random Forests"])

# ==========================================
# DECISION TREES
# ==========================================
with tab1:
    st.header("Decision Trees: The '20 Questions' Game")

    # --- 1. Intuition ---
    st.subheader("1. Intuition: Playing 20 Questions")
    st.markdown(r"""
    Imagine you are playing "20 Questions" to guess if a player will **Win** or **Lose**.
    You want to ask the *best* question first to narrow it down as fast as possible.

    *   **Bad Question**: "Is the wind speed exactly 12.5 km/h?" (Splits data poorly, too specific).
    *   **Good Question**: "Is the Rank Difference > 50?" (Splits data into two clear groups).

    A Decision Tree is just a machine playing this game perfectly.
    """)

    render_mermaid("""
    graph TD
        Root["Is Rank Diff > 50?"] -->|Yes| Left["Is Surface = Clay?"]
        Root -->|No| Right["Is Player Height > 190cm?"]
        Left -->|Yes| L1["Prediction: WIN (90% Prob)"]
        Left -->|No| L2["Prediction: LOSE (80% Prob)"]
        Right -->|Yes| L3["Prediction: WIN (60% Prob)"]
        Right -->|No| L4["Prediction: LOSE (55% Prob)"]

        style Root fill:#fff3e0
        style Left fill:#e3f2fd
        style Right fill:#e3f2fd
        style L1 fill:#c8e6c9
        style L2 fill:#ffcdd2
    """, height=300)

    # --- 2. The Math of Messiness (Walkthrough) ---
    st.subheader("2. How to Choose the Best Question? (Gini Impurity)")
    st.markdown(r"""
    The tree needs a metric to measure "Messiness".
    We use **Gini Impurity**.

    *   **Gini = 0.0**: Perfect Purity (All Wins or All Losses).
    *   **Gini = 0.5**: Maximum Mess (50% Wins, 50% Losses).
    """)

    st.latex(r"Gini = 1 - \sum (p_i)^2")

    st.markdown("### 🧠 Step-by-Step Walkthrough")
    st.markdown("Imagine we have 6 matches at a node:")
    st.code("Matches: [WIN, WIN, WIN, LOSE, LOSE, LOSE]", language="text")
    st.markdown(r"**Current Gini**: $1 - (3/6)^2 - (3/6)^2 = 1 - 0.25 - 0.25 = \mathbf{0.5}$ (Total Mess).")

    st.markdown("---")
    st.markdown("**Option A: Split by 'Rank < 100'**")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Left Child (Rank < 100)**")
        st.code("[WIN, WIN, WIN]", language="text")
        st.markdown(r"Gini = $1 - (3/3)^2 - (0/3)^2 = \mathbf{0.0}$ (Pure!)")
    with col_a2:
        st.markdown("**Right Child (Rank > 100)**")
        st.code("[LOSE, LOSE, LOSE]", language="text")
        st.markdown(r"Gini = $1 - (0/3)^2 - (3/3)^2 = \mathbf{0.0}$ (Pure!)")
    st.success("Weighted Gini = 0.0. This is a PERFECT split!")

    st.markdown("---")
    st.markdown("**Option B: Split by 'Is Sunny?'**")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("**Left Child (Sunny)**")
        st.code("[WIN, LOSE, WIN]", language="text")
        st.markdown(r"Gini = $1 - (2/3)^2 - (1/3)^2 \approx \mathbf{0.44}$")
    with col_b2:
        st.markdown("**Right Child (Not Sunny)**")
        st.code("[LOSE, WIN, LOSE]", language="text")
        st.markdown(r"Gini = $1 - (1/3)^2 - (2/3)^2 \approx \mathbf{0.44}$")
    st.error("Weighted Gini = 0.44. This split barely helped. The tree will choose Option A.")

    # --- 3. Geometry ---
    st.subheader("3. Geometry: The Boxy World")
    st.markdown(r"""
    Because trees ask questions like $x > 5$, they cut the world into **Rectangles (Boxes)**.
    They cannot draw a diagonal line directly. They have to approximate it with a "Staircase".
    """)

    # --- 4. Visualization ---
    st.subheader("4. Interactive Visualization")

    col_viz, col_controls = st.columns([3, 1])
    with col_controls:
        depth = st.slider("Tree Depth", 1, 15, 1)
        dataset = st.selectbox("Dataset", ["Moons", "Circles"], key="tree_data")
        st.caption("Increase Depth to see the 'Staircase' effect on curved data.")

    with col_viz:
        if dataset == "Moons":
            X, y = generate_moons(n_samples=300, noise=0.2)
        else:
            X, y = generate_circles(n_samples=300, noise=0.1)

        clf = DecisionTreeClassifier(max_depth=depth)
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
    st.header("Random Forests: The Wisdom of the Crowd")

    # --- 1. Intuition ---
    st.subheader("1. Intuition: Ask 100 Friends")
    st.markdown(r"""
    A single Decision Tree is like **one smart but erratic expert**.
    *   If you change the data slightly, the tree might change completely (High Variance).
    *   It might memorize the specific noise of the training set (Overfitting).

    **The Solution**: Ask 100 experts.
    *   **Diversity**: Make sure each expert sees slightly different data (Bootstrapping).
    *   **Constraint**: Make sure each expert looks at different clues (Feature Randomness).
    *   **Vote**: Take the majority vote.

    This is called **Bagging** (Bootstrap Aggregating).
    """)

    # --- 2. The Math of Ensembling ---
    st.subheader("2. Why does averaging work?")
    st.markdown(r"""
    If errors are random and uncorrelated, averaging them cancels them out.

    **Variance of the Forest**:
    """)
    st.latex(r"Var(\text{Forest}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2")
    st.markdown(r"""
    *   $\sigma^2$: Variance of one tree (High).
    *   $B$: Number of trees (High is good).
    *   $\rho$: Correlation between trees (Low is good).

    **Key Takeaway**: We want **many** trees ($B \uparrow$) that are **different** from each other ($\rho \downarrow$).
    """)

    # --- 3. Visualization ---
    st.subheader("3. Interactive Visualization")

    col_viz_rf, col_controls_rf = st.columns([3, 1])
    with col_controls_rf:
        n_estimators = st.slider("Num Trees", 1, 100, 10)
        dataset_rf = st.selectbox("Dataset", ["Moons", "Circles"], key="rf_data")

    with col_viz_rf:
        if dataset_rf == "Moons":
            X_rf, y_rf = generate_moons(n_samples=300, noise=0.25)
        else:
            X_rf, y_rf = generate_circles(n_samples=300, noise=0.15)

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

    # --- 7. How to do it in Python ---
st.header("7. How to do it in Python 🐍")
tab_dt_code, tab_rf_code = st.tabs(["Decision Tree", "Random Forest"])

with tab_dt_code:
    st.code("""
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Initialize
dt = DecisionTreeClassifier(max_depth=3, criterion='gini')

# 2. Train
dt.fit(X_train, y_train)

# 3. Predict
y_pred = dt.predict(X_test)

# 4. Visualize
plot_tree(dt, feature_names=feature_names, filled=True)
    """, language="python")

with tab_rf_code:
    st.code("""
from sklearn.ensemble import RandomForestClassifier

# 1. Initialize
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# 2. Train
rf.fit(X_train, y_train)

# 3. Predict
y_prob = rf.predict_proba(X_test)[:, 1]

# 4. Feature Importance
print(rf.feature_importances_)
    """, language="python")

# --- 8. Super Summary ---
    st.subheader("8. Super Summary 🦸")
    st.info(r"""
    *   **Decision Tree**: A set of If-Then rules (20 Questions).
    *   **Gini Impurity**: The math behind "Good Questions".
    *   **Random Forest**: A democracy of trees.
    *   **Key Insight**: Averaging many "okay" models creates one "super" model.
""")
