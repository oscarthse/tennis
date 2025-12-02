import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.model_cards import render_model_card
from src.dashboard.components.toy_datasets import generate_moons, generate_circles
from src.dashboard.components.mermaid import render_mermaid

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
""")

st.markdown("---")

# 3. The Toy Dataset
st.header("2. The Toy Dataset")
st.markdown("To understand how the tree is built, let's look at a tiny dataset of **6 Matches**.")

toy_data = pd.DataFrame({
    'Match': [1, 2, 3, 4, 5, 6],
    'RankDiff': [10, -5, 20, 50, -2, 5],
    'Surface': ['Clay', 'Grass', 'Clay', 'Hard', 'Grass', 'Hard'],
    'Winner': ['Win', 'Win', 'Lose', 'Lose', 'Win', 'Lose']
})

st.dataframe(toy_data, use_container_width=True)

st.markdown("""
*   **RankDiff**: (Player Rank - Opponent Rank). Positive means Player is worse.
*   **Goal**: Separate the **3 Wins** from the **3 Losses**.
""")

st.markdown("---")

# 4. Building the Tree (Step-by-Step)
st.header("3. Building the Tree: Step-by-Step")
st.markdown("The tree wants to find the question that makes the resulting groups as **pure** as possible.")

st.subheader("Step 1: Calculate Impurity (Gini) of the Root")
st.markdown("At the start (Root Node), we have all 6 matches: **3 Wins, 3 Losses**.")
st.markdown("The **Gini Impurity** formula is:")
st.latex(r"G = 1 - \sum p_i^2")

st.markdown("""
*   $p_{win} = 3/6 = 0.5$
*   $p_{loss} = 3/6 = 0.5$
""")
st.latex(r"G_{root} = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = 0.5")
st.markdown("**Gini = 0.5** means maximum impurity (total chaos). We want to get this to 0.0.")

st.subheader("Step 2: Try a Split on 'RankDiff < 8'")
st.markdown("Let's ask: *Is RankDiff less than 8?*")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Yes (Left Node)**")
    st.markdown("Matches: 2, 5, 6")
    st.markdown("*   RankDiffs: -5, -2, 5")
    st.markdown("*   Results: **Win, Win, Lose**")
    st.markdown("*   **2 Wins, 1 Loss**")
    st.latex(r"G_{left} = 1 - ((\frac{2}{3})^2 + (\frac{1}{3})^2) \approx 0.44")

with col2:
    st.markdown("**No (Right Node)**")
    st.markdown("Matches: 1, 3, 4")
    st.markdown("*   RankDiffs: 10, 20, 50")
    st.markdown("*   Results: **Win, Lose, Lose**")
    st.markdown("*   **1 Win, 2 Losses**")
    st.latex(r"G_{right} = 1 - ((\frac{1}{3})^2 + (\frac{2}{3})^2) \approx 0.44")

st.markdown("**Weighted Gini for this Split:**")
st.latex(r"G_{split} = \frac{3}{6}(0.44) + \frac{3}{6}(0.44) = 0.44")
st.markdown("Improvement: $0.5 - 0.44 = 0.06$. (A small gain).")

st.subheader("Step 3: Try a Split on 'Surface == Clay'")
st.markdown("Let's ask: *Is the Surface Clay?*")

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Yes (Left Node)**")
    st.markdown("Matches: 1, 3")
    st.markdown("*   Results: **Win, Lose**")
    st.markdown("*   **1 Win, 1 Loss**")
    st.latex(r"G_{left} = 0.5 \text{ (Still impure)}")

with col4:
    st.markdown("**No (Right Node)**")
    st.markdown("Matches: 2, 4, 5, 6")
    st.markdown("*   Results: **Win, Lose, Win, Lose**")
    st.markdown("*   **2 Wins, 2 Losses**")
    st.latex(r"G_{right} = 0.5 \text{ (Still impure)}")

st.markdown("**Weighted Gini:** 0.5. **Improvement: 0.0**. This split is useless!")

st.subheader("Step 4: The Best Split")
st.markdown("The computer tries *every possible number*. It finds that **RankDiff < 0** is the best split.")
st.markdown("*   **Yes**: Matches 2, 5 (-5, -2). **2 Wins, 0 Losses**. (Pure!)")
st.markdown("*   **No**: Matches 1, 3, 4, 6. **1 Win, 3 Losses**.")

st.markdown("This reduces impurity the most. So the tree picks this as the Root Question.")

st.subheader("The Final Tree")
render_mermaid("""
graph TD
    Root["RankDiff < 0? <br> (3 Win, 3 Loss)"] -->|Yes| Left["Leaf: WIN <br> (2 Win, 0 Loss) <br> Gini=0.0"]
    Root -->|No| Right["RankDiff < 30? <br> (1 Win, 3 Loss)"]
    Right -->|Yes| R_Left["Leaf: LOSE <br> (0 Win, 3 Loss)"]
    Right -->|No| R_Right["Leaf: WIN <br> (1 Win, 0 Loss)"]

    style Left fill:#c8e6c9
    style R_Left fill:#ffcdd2
    style R_Right fill:#c8e6c9
""", height=400)

st.markdown("---")

# 5. Random Forests
st.header("4. Random Forests: The Wisdom of Crowds")
st.markdown("""
Decision Trees have a fatal flaw: **They memorize data.**
If we changed one match in our dataset, the whole tree might change. This is called **High Variance**.

To fix this, we don't just build one tree. We build **100 trees** and let them vote.
This is a **Random Forest**.

### The Math: Why Averaging Works
Why does averaging 100 bad trees make one good forest?
The variance of the average of $n$ trees is:

$$
Var(\text{Forest}) = \rho \sigma^2 + \frac{1-\rho}{n} \sigma^2
$$

*   $\sigma^2$: Variance of a single tree (High).
*   $n$: Number of trees (e.g., 100).
*   $\rho$ (rho): **Correlation** between trees.

**The Goal**:
1.  Increase $n$ (More trees) $\to$ Second term vanishes.
2.  **Decrease $\rho$** (Make trees different) $\to$ First term shrinks.

**How do we decrease correlation ($\rho$)?**
We force the trees to be different using **Bootstrapping** and **Feature Randomness**.
""")

st.subheader("Secret Sauce 1: Bootstrapping (Bagging)")
st.markdown("If we gave the same dataset to 100 trees, they would all look identical. We need them to be different.")
st.markdown("We create **Bootstrap Samples** (Random sampling *with replacement*).")

col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    st.markdown("**Tree 1's Data**")
    st.markdown("Match 1, 1, 2, 5, 6, 6")
    st.caption("Notice duplicates and missing Match 3 & 4.")
with col_b2:
    st.markdown("**Tree 2's Data**")
    st.markdown("Match 2, 3, 3, 4, 5, 6")
with col_b3:
    st.markdown("**Tree 3's Data**")
    st.markdown("Match 1, 2, 3, 4, 4, 5")

st.subheader("Secret Sauce 2: Feature Randomness")
st.markdown("""
At every split, the tree is **NOT allowed to look at all features**.
It is forced to pick from a random subset (e.g., only 'Surface' and 'Odds').
*   This forces trees to learn different patterns.
*   It prevents one strong feature (like RankDiff) from dominating every tree.
""")

st.subheader("The Forest Vote")
render_mermaid("""
graph LR
    Input["New Match: <br> RankDiff=5, Surface=Clay"] --> T1
    Input --> T2
    Input --> T3

    subgraph Forest
        T1["Tree 1"] -->|Vote| V1["Win"]
        T2["Tree 2"] -->|Vote| V2["Lose"]
        T3["Tree 3"] -->|Vote| V3["Win"]
    end

    V1 --> Final["Majority Vote: <br> WIN"]
    V2 --> Final
    V3 --> Final
""", height=300)

st.markdown("---")

# 6. Interactive Viz
st.header("5. Interactive Visualization")
st.markdown("Compare a single Decision Tree vs. a Random Forest.")

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

    if max_depth > 10 and model_type == "Decision Tree":
        st.warning("⚠️ **Overfitting Alert**: Notice how the Decision Tree creates tiny, jagged islands to capture every single noise point? This is bad for future data.")
    if model_type == "Random Forest":
        st.success("✅ **Smoother Boundary**: The Random Forest averages out the jagged edges, creating a more robust model.")

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Try Random Forest in the Playground", icon="🎮")
