import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.model_cards import render_model_card
from src.dashboard.components.toy_datasets import generate_moons, generate_circles

st.set_page_config(page_title="Trees & Forests", page_icon="🌳", layout="wide")
sidebar_navigation()

st.title("🌳 Decision Trees & Random Forests")

# 1. TL;DR
render_model_card(
    title="Decision Tree",
    description="A non-linear model that makes decisions by asking a series of Yes/No questions (splits) to separate the data. It mimics human decision-making.",
    pros=["Highly interpretable (white-box model)", "Handles non-linear data well", "No feature scaling needed"],
    cons=["Prone to overfitting (memorizing noise)", "Unstable (small data changes = different tree)"]
)

st.markdown("---")

# 2. Intuition
st.header("1. Intuition: The Game of 20 Questions")
st.markdown("""
### The Tennis Analogy
Imagine you are a commentator trying to guess if a player will win. You can only ask Yes/No questions.

*   **Question 1**: "Is the player ranked in the Top 10?"
    *   **No**: (Most likely loses).
    *   **Yes**: (Okay, good chance).
*   **Question 2 (if Yes)**: "Is the opponent Djokovic?"
    *   **Yes**: (Probably loses).
    *   **No**: (Probably wins).

A **Decision Tree** automates this. It looks at all possible questions (features) and picks the one that **best separates** the Winners from the Losers.

*   **Root Node**: The first question (e.g., Rank).
*   **Leaf Node**: The final prediction (Win/Loss).
*   **Depth**: How many questions we ask.
""")

st.markdown("---")

# 3. The Math
st.header("2. The Math: Measuring 'Purity'")
st.markdown("How does the tree decide which question is 'best'? It tries to maximize **Information Gain**. It wants the groups after the split to be as **pure** as possible (all Wins or all Losses).")

st.subheader("Entropy (The Measure of Chaos)")
st.markdown("Entropy $H(S)$ measures the uncertainty in a dataset $S$.")
st.latex(r"H(S) = - \sum_{i=1}^c p_i \log_2(p_i)")

st.markdown("""
*   $p_i$: The proportion of class $i$ in the node.
*   **Max Entropy (1.0)**: 50% Wins, 50% Losses. (Total Chaos).
*   **Min Entropy (0.0)**: 100% Wins. (Total Order).
""")

st.subheader("Gini Impurity (The Alternative)")
st.markdown("Gini is faster to compute (no logs) and is the default in Scikit-Learn.")
st.latex(r"G(S) = 1 - \sum_{i=1}^c p_i^2")
st.markdown("Like Entropy, it is 0 for pure nodes and maximum (0.5) for 50/50 mixes.")

st.subheader("Information Gain (The Decision Rule)")
st.markdown("The tree picks the split that reduces Entropy the most.")
st.latex(r"IG(S, A) = H(S) - \sum_{v \in \text{Children}} \frac{|S_v|}{|S|} H(S_v)")
st.markdown("*   $H(S)$: Entropy before split.\n*   Weighted Sum: Average entropy after split.")

st.markdown("---")

# 4. Worked Example
st.header("3. Step-by-Step Worked Example")
st.markdown("Let's calculate the Entropy of a node manually.")

st.markdown("### The Node")
st.markdown("Suppose we have a group of **10 matches**:")
st.markdown("*   **6 Wins (+)**")
st.markdown("*   **4 Losses (-)**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Step 1: Calculate Proportions")
    st.latex(r"p_+ = \frac{6}{10} = 0.6")
    st.latex(r"p_- = \frac{4}{10} = 0.4")

with col2:
    st.markdown("#### Step 2: Calculate Entropy")
    st.latex(r"H(S) = - [0.6 \log_2(0.6) + 0.4 \log_2(0.4)]")
    st.markdown("Using $\log_2(0.6) \approx -0.737$ and $\log_2(0.4) \approx -1.322$:")
    st.latex(r"H(S) = - [0.6(-0.737) + 0.4(-1.322)]")
    st.latex(r"H(S) = - [-0.442 - 0.529] = 0.971")

st.success("Result: Entropy is **0.971**. This is very high (close to 1), meaning the group is very mixed/impure.")

st.markdown("#### Step 3: Evaluate a Split")
st.markdown("Now imagine we split this group by **Surface=Clay**.")
st.markdown("*   **Left Child (Clay)**: 5 Wins, 0 Losses. (Pure!)")
st.markdown("*   **Right Child (Hard)**: 1 Win, 4 Losses.")

st.markdown("**Entropy(Left)** = 0.0 (Pure).")
st.markdown("**Entropy(Right)** = Calculate for (1/5, 4/5)... it will be lower than 0.971.")
st.markdown("The weighted average will be much lower than 0.971, so **Information Gain is High**. This is a good split!")

st.markdown("---")

# 5. Code
st.header("4. The Code")
st.code("""
from sklearn.tree import DecisionTreeClassifier

# 1. Initialize
# max_depth=3: Only ask 3 questions (prevents overfitting)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# 2. Train
model.fit(X_train, y_train)

# 3. Visualize
from sklearn.tree import plot_tree
plot_tree(model, feature_names=['RankDiff', 'PointsDiff'])
""", language="python")

st.markdown("---")

# 6. Interactive Viz
st.header("5. Interactive Visualization")
st.markdown("Compare a single Decision Tree vs. a Random Forest (Bagging).")

col1, col2 = st.columns([1, 3])

with col1:
    dataset_type = st.selectbox("Dataset", ["Moons", "Circles"])
    noise = st.slider("Noise", 0.0, 1.0, 0.3)

    model_type = st.radio("Model", ["Decision Tree", "Random Forest"])
    max_depth = st.slider("Max Depth", 1, 20, 5, help="Deeper trees fit data better but can overfit.")

    if model_type == "Random Forest":
        n_estimators = st.slider("Number of Trees", 1, 100, 10)
    else:
        n_estimators = 1

with col2:
    # Generate Data
    if dataset_type == "Moons":
        X, y = generate_moons(noise=noise)
    else:
        X, y = generate_circles(noise=noise)

    # Train Model
    if model_type == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    clf.fit(X, y)

    # Plot Decision Boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    # Contour (Class Regions)
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.02),
        y=np.arange(y_min, y_max, 0.02),
        z=Z,
        colorscale='RdBu',
        opacity=0.4,
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

    fig.update_layout(title=f"{model_type} Boundary (Acc: {clf.score(X, y):.2f})", height=500)
    st.plotly_chart(fig, use_container_width=True)

    if max_depth > 10:
        st.warning("⚠️ High depth! Notice how the boundary becomes jagged and captures noise? This is **Overfitting**.")
    if model_type == "Random Forest":
        st.success("✅ Random Forest averages many trees to create a smoother, more robust boundary.")

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Try Random Forest in the Playground", icon="🎮")
