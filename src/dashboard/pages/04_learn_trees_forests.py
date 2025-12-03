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

tab1, tab_entropy, tab2 = st.tabs(["Decision Trees", "Entropy Deep Dive", "Random Forests"])

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
    The tree needs a metric to measure "Messiness". We use **Gini Impurity**.

    **Intuition: The "Party Game"** 🎈
    Imagine you are at a party with people from Team A and Team B.
    If you pick two people at random, **what is the probability they belong to DIFFERENT teams?**

    *   **Everyone is Team A**: Probability is **0**. (Pure).
    *   **50/50 Split**: Probability is **0.5**. (Maximum Mess).

    **The Formula**:
    $$Gini = 1 - \sum (p_i)^2$$

    *   $\sum (p_i)^2$ is the probability of picking the **SAME** class twice.
    *   $1 - \dots$ is the probability of picking **DIFFERENT** classes.
    """)

    st.markdown("### 🧠 Step-by-Step Walkthrough")
    st.markdown("Let's calculate the **Gini Gain** for a split. This is how the tree decides.")

    st.markdown("**Step 1: The Parent Node**")
    st.code("Matches: [WIN, WIN, WIN, LOSE, LOSE, LOSE] (Total 6)", language="text")
    st.markdown(r"""
    *   $p(Win) = 3/6 = 0.5$
    *   $p(Lose) = 3/6 = 0.5$
    *   $Gini(Parent) = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = \mathbf{0.5}$
    """)

    st.markdown("---")
    st.markdown("**Step 2: The Split (Option A: Rank < 100)**")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Left Child (3 items)**")
        st.code("[WIN, WIN, WIN]", language="text")
        st.markdown(r"""
        *   $p(W)=1.0, p(L)=0.0$
        *   $Gini = 1 - (1^2 + 0^2) = \mathbf{0.0}$
        """)
    with col_a2:
        st.markdown("**Right Child (3 items)**")
        st.code("[LOSE, LOSE, LOSE]", language="text")
        st.markdown(r"""
        *   $p(W)=0.0, p(L)=1.0$
        *   $Gini = 1 - (0^2 + 1^2) = \mathbf{0.0}$
        """)

    st.markdown("**Step 3: Weighted Average & Gain**")
    st.latex(r"Weighted Gini = \frac{3}{6}(0.0) + \frac{3}{6}(0.0) = \mathbf{0.0}")
    st.latex(r"Gini Gain = Gini(Parent) - Weighted Gini = 0.5 - 0.0 = \mathbf{0.5}")
    st.success("This is a massive gain! The tree loves this split.")

    st.markdown("---")
    st.markdown("**Step 4: A Bad Split (Option B: Is Sunny?)**")
    st.markdown("Imagine a split that separates them into `[WIN, WIN, LOSE]` and `[WIN, LOSE, LOSE]`.")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("**Left Child (3 items)**")
        st.code("[WIN, WIN, LOSE]", language="text")
        st.markdown(r"""
        *   $p(W)=0.67, p(L)=0.33$
        *   $Gini = 1 - (0.67^2 + 0.33^2) \approx \mathbf{0.44}$
        """)
    with col_b2:
        st.markdown("**Right Child (3 items)**")
        st.code("[WIN, LOSE, LOSE]", language="text")
        st.markdown(r"""
        *   $p(W)=0.33, p(L)=0.67$
        *   $Gini = 1 - (0.33^2 + 0.67^2) \approx \mathbf{0.44}$
        """)

    st.latex(r"Weighted Gini = \frac{3}{6}(0.44) + \frac{3}{6}(0.44) = \mathbf{0.44}")
    st.latex(r"Gini Gain = 0.5 - 0.44 = \mathbf{0.06}")
    st.error("Gain is tiny (0.06). The tree will REJECT this split and choose Option A.")

    # --- 3. Geometry ---
    st.subheader("3. Geometry: The Boxy World")
    st.markdown(r"""
    Because trees ask questions like $x > 5$, they cut the world into **Rectangles (Boxes)**.
    They cannot draw a diagonal line directly. They have to approximate it with a "Staircase".
    """)

    # --- 4. The Algorithm: Under the Hood ---
    st.subheader("4. The Algorithm: Under the Hood ⚙️")
    st.markdown(r"""
    We have seen the *math* (Gini). Now let's look at the *code* (The CART Algorithm).
    **CART** = Classification And Regression Trees.

    This is a recursive, greedy algorithm. It builds the tree top-down.
    """)

    st.markdown("### 4.1 The Training Loop (Pseudocode)")
    st.code("""
def BuildTree(data):
    # 1. Check Stopping Rules
    if (Pure Node) or (Max Depth Reached) or (Too Few Samples):
        return CreateLeaf(data)

    # 2. Find Best Split
    best_gain = 0
    best_question = None

    for feature in all_features:
        # Try every possible threshold!
        thresholds = GetUniqueThresholds(data[feature])

        for t in thresholds:
            gain = CalculateGiniGain(data, feature, t)
            if gain > best_gain:
                best_gain = gain
                best_question = (feature, t)

    # 3. Recurse
    left_data, right_data = Split(data, best_question)

    node = Node(question=best_question)
    node.left = BuildTree(left_data)   # Magic happens here!
    node.right = BuildTree(right_data)

    return node
    """, language="python")

    st.markdown("#### 🚶‍♂️ Walking Through the Code")
    st.markdown("""
    Let's trace exactly what happens when we call `BuildTree(data)`:

    **1. The Safety Check (Stopping Rules)**
    *   First, the function looks around. "Is everyone here already a Winner?" (Pure Node). "Am I too deep?" (Max Depth).
    *   If **YES**, it stops immediately. It creates a **Leaf Node** (a final prediction) and returns it. No more recursion.

    **2. The Great Search (Finding the Best Split)**
    *   If we didn't stop, we need to split.
    *   The algorithm is **Greedy**. It loops through **Every Feature** (Rank, Surface, Height...).
    *   Inside that, it loops through **Every Possible Threshold** (Rank < 10, Rank < 20...).
    *   It calculates the **Gini Gain** for *thousands* of possibilities and keeps the single best one (e.g., `Rank < 50`).

    **3. The Magic (Recursion)**
    *   Now we have a question: `Rank < 50`.
    *   We physically split the data into two piles: `left_data` (Rank < 50) and `right_data` (Rank >= 50).
    *   **Here is the magic**: We call `BuildTree(left_data)`.
    *   The function **pauses**, goes into a new dimension with only the Left Data, and starts from Step 1 again.
    *   When that finishes, it calls `BuildTree(right_data)`.
    *   Finally, it connects these two new sub-trees to itself and returns the full structure.
    """)

    st.markdown("### 4.2 Handling Numeric Features (The Sorting Trick)")
    st.markdown("""
    How does the tree know to split at `Rank > 50`? Why not 51? Why not 49.5?

    **The Algorithm:**
    1.  **Sort** the unique values of the feature.
    2.  **Generate Candidates**: Take the midpoint between every adjacent pair.
    3.  **Test**: Calculate Gini for *every single candidate*.
    """)

    st.markdown("**Example: Player Rank**")
    st.code("Data: [10, 20, 50, 80]", language="text")
    st.markdown("""
    *   **Midpoint 1**: (10+20)/2 = **15**
    *   **Midpoint 2**: (20+50)/2 = **35**
    *   **Midpoint 3**: (50+80)/2 = **65**

    The tree tests `Rank < 15`, `Rank < 35`, and `Rank < 65`. It picks the winner.
    """)

    st.info("""
    **Optimization Bam!** 💥
    Naive implementations recalculate Gini from scratch for every threshold ($O(N^2)$).
    Professional implementations (LightGBM, XGBoost) use **Histograms**.
    They move one point from the "Right Bin" to the "Left Bin" and update the counts in $O(1)$.
    """)

    st.markdown("### 4.3 Handling Categorical Features")
    st.markdown("""
    **Binary (Yes/No)**: Easy. `Is Surface == Clay?`

    **Multi-Class (A, B, C)**:
    *   **One-vs-Rest**: `Is Surface == A?` vs `Is Surface != A?`
    *   **Subset Split**: `Is Surface in {A, C}?` (More complex, tries combinations).
    """)

    st.markdown("### 4.4 Recursion & Data Flow 🔄")
    st.markdown("""
    This is where the magic happens.
    1.  **Root Node**: Receives 1000 samples. Finds best split.
    2.  **Split**: Sends 600 samples to Left Child, 400 to Right Child.
    3.  **Left Child**: "I am now the Root of my own tiny world." I repeat the process on my 600 samples.
    4.  **Right Child**: Does the same.

    This continues until a **Stopping Rule** is hit.
    """)

    st.markdown("### 4.5 Stopping Rules (The Brakes 🛑)")
    st.markdown("""
    If we don't stop, the tree will grow until every single leaf has 1 sample (Overfitting).

    1.  **`max_depth`**: "Stop if you are 5 levels deep."
    2.  **`min_samples_split`**: "Stop if you have fewer than 20 samples."
    3.  **`min_impurity_decrease`**: "Stop if the best split only improves Gini by 0.0001."
    4.  **Pure Node**: "Stop if everyone here is a Winner (Gini=0)."
    """)

    st.markdown("### 4.6 Leaf Creation 🍃")
    st.markdown("""
    When a node stops, it becomes a **Leaf**.
    It looks at the samples remaining in its bucket.

    *   **Classification**: Returns the **Majority Class** (e.g., "Win").
    *   **Probabilities**: Returns the ratio (e.g., "80% Win, 20% Lose").
    """)

    # --- 5. Visualization ---
    st.subheader("5. Interactive Visualization")

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
# ENTROPY DEEP DIVE
# ==========================================
with tab_entropy:
    st.header("Entropy: The measure of Surprise! 😲")

    # --- 1. What is Entropy? ---
    st.subheader("1. What is Entropy? (The Surprise!)")
    st.markdown(r"""
    **Entropy is simply the average amount of 'Surprise' inherent in a dataset.**

    Let's build this intuition from scratch. Imagine you are watching cars go by.
    *   **Scenario A**: You see a Toyota (Very Common). Are you surprised? **No.** (Low Surprise).
    *   **Scenario B**: You see a Ferrari (Very Rare). Are you surprised? **Yes!** (High Surprise).

    **Bam!** Insight #1: **Surprise is inversely related to Probability.**
    *   High Probability ($p \approx 1$) $\rightarrow$ Low Surprise ($\approx 0$).
    *   Low Probability ($p \approx 0$) $\rightarrow$ High Surprise ($\approx \infty$).
    """)

    st.markdown("### Deriving the Formula for Surprise")
    st.markdown(r"""
    We need a math function that behaves like this.

    **Attempt 1**: $\frac{1}{p}$
    *   If $p=1$, Surprise = 1. (Close, but we want 0 surprise for certainty).
    *   If $p=0.1$, Surprise = 10. (Okay).

    **Attempt 2**: $\log(\frac{1}{p})$
    *   If $p=1$, $\log(1) = 0$. **Perfect!** Certainty means zero surprise.
    *   If $p \rightarrow 0$, $\log(\text{huge}) \rightarrow \infty$. **Perfect!** Impossible events are infinitely surprising.

    We use $\log_2$ (base 2) because in computer science, we measure information in **Bits**.
    """)

    st.latex(r"Surprise(x) = \log_2\left(\frac{1}{p(x)}\right) = -\log_2(p(x))")

    # --- 2. From Surprise to Entropy ---
    st.subheader("2. From Surprise to Entropy")
    st.markdown(r"""
    So, **Entropy** is just the **Expected Value** of the Surprise.
    It answers: *"On average, how surprised am I going to be by the next data point?"*
    """)

    st.latex(r"""
    H(X) = \sum p(x) \cdot Surprise(x) \\
    H(X) = \sum p(x) \cdot \left( -\log_2(p(x)) \right) \\
    \boxed{H(X) = - \sum p(x) \log_2(p(x))}
    """)

    st.info("This is **Shannon's Entropy Formula**. Elegant, isn't it?")

    # --- 3. The Chicken Analogy ---
    st.subheader("3. The Chicken Analogy 🐔")
    st.markdown("""
    Let's apply this to three areas with Orange and Blue chickens.
    We want to know which area is the "Messiest" (Highest Entropy).
    """)

    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        st.markdown("**Area A (Clean)**")
        st.write("🟠🟠🟠🟠🟠🔵")
        st.caption("6 Orange, 1 Blue")
        st.markdown(r"""
        $p(O) \approx 0.86, p(B) \approx 0.14$
        **Entropy is Low.**
        Most of the time, you see Orange. You are rarely surprised.
        """)

    with col_c2:
        st.markdown("**Area B (Clean)**")
        st.write("🔵🔵🔵🔵🔵🟠")
        st.caption("1 Orange, 6 Blue")
        st.markdown(r"""
        $p(O) \approx 0.14, p(B) \approx 0.86$
        **Entropy is Low.**
        Most of the time, you see Blue. You are rarely surprised.
        """)

    with col_c3:
        st.markdown("**Area C (Messy)**")
        st.write("🟠🟠🟠🔵🔵🔵")
        st.caption("3 Orange, 3 Blue")
        st.markdown(r"""
        $p(O) = 0.5, p(B) = 0.5$
        **Entropy is High (Max).**
        You have NO IDEA what comes next. Maximum unpredictability.
        """)

    # --- 4. Entropy vs Gini ---
    st.subheader("4. Entropy vs Gini Impurity 🥊")
    st.markdown(r"""
    Both metrics measure "Messiness", but they come from different places.

    *   **Entropy**: From Information Theory (Bits, Surprise).
    *   **Gini**: From Economics/Statistics (Probability of misclassification).

    **The Comparison**:
    """)

    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.markdown("**Entropy**")
        st.latex(r"H = -\sum p_i \log_2(p_i)")
        st.write("- Range: [0, 1] (for binary)")
        st.write("- Slower to compute (logs)")
        st.write("- Slightly more sensitive to changes")

    with col_comp2:
        st.markdown("**Gini**")
        st.latex(r"G = 1 - \sum p_i^2")
        st.write("- Range: [0, 0.5] (for binary)")
        st.write("- Faster to compute (squares)")
        st.write("- The default in sklearn")

    # Plotting Gini vs Entropy
    p = np.linspace(0.001, 0.999, 100)
    entropy = - (p * np.log2(p) + (1-p) * np.log2(1-p))
    gini = 1 - (p**2 + (1-p)**2)
    # Scale Gini to match Entropy height for comparison
    gini_scaled = gini * 2

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=p, y=entropy, mode='lines', name='Entropy', line=dict(color='blue')))
    fig_comp.add_trace(go.Scatter(x=p, y=gini, mode='lines', name='Gini (Raw)', line=dict(color='green', dash='dash')))
    fig_comp.add_trace(go.Scatter(x=p, y=gini_scaled, mode='lines', name='Gini (Scaled x2)', line=dict(color='red', dash='dot')))

    fig_comp.update_layout(
        title="Entropy vs Gini (Binary Classification)",
        xaxis_title="Probability of Class 1 (p)",
        yaxis_title="Impurity Value",
        height=400
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption("Notice how similar the shapes are! This is why they often yield similar results.")

    # --- 5. Calculation Lab ---
    st.subheader("5. Calculation Lab 🧮")
    st.markdown("Let's calculate the **Information Gain** for a split manually.")

    st.markdown("**Parent Node**: 4 Cats 🐱, 6 Dogs 🐶 (Total 10)")
    st.markdown(r"""
    1.  **Calculate Parent Entropy**:
        *   $p(Cat) = 0.4$, $p(Dog) = 0.6$
        *   $H(Parent) = -(0.4 \log_2 0.4 + 0.6 \log_2 0.6)$
        *   $H(Parent) \approx -(0.4 \cdot -1.32 + 0.6 \cdot -0.737) \approx \mathbf{0.971}$
    """)

    st.markdown("---")
    st.markdown("**The Split**: We split by 'Likes Meow Mix?'")

    col_lab1, col_lab2 = st.columns(2)
    with col_lab1:
        st.markdown("**Left Child (Yes)**")
        st.write("🐱🐱🐱🐱 (4 Cats, 0 Dogs)")
        st.markdown(r"""
        *   $p(Cat)=1.0, p(Dog)=0.0$
        *   $H(Left) = -(1 \log 1 + 0) = \mathbf{0.0}$
        *   (Pure node!)
        """)

    with col_lab2:
        st.markdown("**Right Child (No)**")
        st.write("🐶🐶🐶🐶🐶🐶 (0 Cats, 6 Dogs)")
        st.markdown(r"""
        *   $p(Cat)=0.0, p(Dog)=1.0$
        *   $H(Right) = \mathbf{0.0}$
        *   (Pure node!)
        """)

    st.markdown("### Information Gain")
    st.latex(r"Gain = H(Parent) - \text{Weighted Average}(H(Children))")
    st.markdown(r"""
    $Gain = 0.971 - ( \frac{4}{10} \cdot 0 + \frac{6}{10} \cdot 0 ) = \mathbf{0.971}$

    **Bam!** This is a massive gain. The tree would definitely choose this split.
    """)

    # --- 6. Mastery Checklist ---
    st.subheader("6. Mastery Checklist & Exercises ✅")
    st.markdown("""
    - [ ] **Explain Surprise**: Why is $\log(1/p)$ the right function?
    - [ ] **Define Entropy**: Expected value of surprise.
    - [ ] **Compare**: Why does a 50/50 split have max entropy?
    - [ ] **Calculate**: Can you compute entropy for [3, 3]? (Hint: It's 1.0).
    - [ ] **Gain**: How does entropy help the tree decide?
    """)

    with st.expander("📝 Practice Problems"):
        st.markdown("""
        1.  **Compute Entropy** for a node with [9 Wins, 1 Loss].
        2.  **Compare**: Which has higher entropy: [5, 5] or [6, 4]?
        3.  **Gini vs Entropy**: Calculate both for [10, 0].
        4.  **Intuition**: Explain to a 5-year-old why we use logs.
        5.  **Scenario**: You have 3 classes [3, 3, 3]. What is the entropy? (Hint: $>1$).
        """)

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

    # --- 3. The Algorithm: Step-by-Step ---
    st.subheader("3. The Algorithm: Step-by-Step ⚙️")
    st.markdown(r"""
    How do we actually build this "Forest"? It's a 4-step recipe.

    **Step 1: Bootstrapping (The Data)** 🎲
    We don't train every tree on the same data. We create a **Bootstrap Sample**.
    *   Pick a sample from the dataset.
    *   Put it back (Replacement).
    *   Pick another.
    *   Repeat until you have a dataset of the same size ($N$).

    *Result*: Some samples are picked twice (duplicates), some are never picked (**Out-of-Bag**).
    """)

    st.code("""
Original: [A, B, C, D]
Bootstrap 1: [A, B, B, D]  (C is missing!)
Bootstrap 2: [A, C, D, D]  (B is missing!)
    """, language="text")

    st.markdown(r"""
    **Step 2: Feature Randomness (The Split)** 🙈
    When a tree wants to split a node, it is **NOT allowed** to look at all features.
    *   It must pick a random subset of features (usually $\sqrt{\text{Total Features}}$).
    *   It finds the best split *only among those few features*.

    *Why?* If there is one "Super Feature" (e.g., Rank), every single tree would use it at the top. They would all look identical.
    By forcing them to ignore it sometimes, we find other subtle patterns.
    """)

    st.markdown(r"""
    **Step 3: Grow the Tree** 🌳
    Grow the tree deep! No pruning. Let it overfit its weird little bootstrap dataset.

    **Step 4: Repeat** 🔁
    Do this 100 or 1000 times.
    """)

    st.markdown("### 🧱 Concrete Example")
    st.markdown("""
    **Dataset**: 4 Matches (M1, M2, M3, M4). Features: Rank, Surface, Wind.

    **Tree 1**:
    1.  **Bootstrap**: `[M1, M2, M2, M4]`.
    2.  **Root Split**: Randomly selects `[Surface, Wind]`. Ignores Rank.
    3.  **Best Split**: `Surface == Clay`.

    **Tree 2**:
    1.  **Bootstrap**: `[M1, M3, M4, M4]`.
    2.  **Root Split**: Randomly selects `[Rank, Wind]`. Ignores Surface.
    3.  **Best Split**: `Rank < 50`.

    **Prediction**:
    New Match comes in.
    *   Tree 1 says: **WIN**
    *   Tree 2 says: **LOSE**
    *   Tree 3 says: **WIN**
    *   **Forest says**: **WIN** (2 vs 1).
    """)

    st.info("""
    **Bonus: Out-of-Bag (OOB) Error** 🎒
    Remember sample 'C' that was left out of Tree 1?
    We can use Tree 1 to predict 'C' and see if it's right.
    If we do this for every tree and every sample, we get a **Validation Score for FREE** without needing a separate test set!
    """)

    # --- 4. Visualization ---
    st.subheader("4. Interactive Visualization")

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
