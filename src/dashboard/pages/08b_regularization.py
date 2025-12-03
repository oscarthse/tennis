import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.dashboard.components.navigation import sidebar_navigation

st.set_page_config(page_title="Regularization (L1 & L2)", page_icon="🛡️", layout="wide")
sidebar_navigation()

st.title("🛡️ Regularization: The Art of Restraint (L1 & L2)")
st.markdown("**Elevator Pitch:** Prevent your model from \"memorizing\" the training data by punishing it for being too complex.")

st.markdown("### Learning Outcomes")
st.markdown("""
*   Understand the difference between **Memorizing** (Overfitting) and **Learning** (Generalization).
*   Learn how **L2 (Ridge)** acts like a rubber band to shrink weights.
*   Learn how **L1 (Lasso)** acts like a budget to select features.
*   Visualize the effect of the **C parameter** on the decision boundary.
""")

# --- 1. Motivation ---
st.header("1. Motivation: The \"Memorizing\" Player 🧠")
st.markdown("""
Imagine two tennis players analyzing their opponent, **Roger**:

1.  **Player A (The Learner)**: Notices that Roger hits a cross-court forehand 80% of the time when he is pulled wide.
    *   *Rule*: "If Roger is wide -> Expect Cross-Court."
    *   *Result*: Simple, robust, works in most matches.

2.  **Player B (The Memorizer)**: Memorizes every single point from the last match.
    *   *Rule*: "If Roger is wide, AND it's 30-15, AND he's wearing a green shirt, AND the wind is 5mph -> Expect Drop Shot."
    *   *Result*: This rule worked perfectly *in the past match*, but it will fail miserably today.

**Player B is Overfitting.** They have learned the *noise*, not the *signal*.
""")

st.subheader("The Math of Overfitting")
st.markdown(r"""
In Logistic Regression, we minimize the **Loss Function** $J(w, b)$.
If we let the weights $w$ get infinitely large, the model can create incredibly complex, wiggly boundaries to fit every single outlier.

**Regularization** adds a "Penalty" term to the Loss function to keep the weights small.
""")

# --- 2. L2 Regularization ---
st.header("2. L2 Regularization (Ridge) ⭕")
st.markdown(r"""
**The Rubber Band.**

We add the **Sum of Squared Weights** to the loss function.
""")
st.latex(r"J_{regularized}(w) = J(w) + \lambda \sum_{j=1}^n w_j^2")
st.markdown(r"""
*   **$\lambda$ (Lambda)**: The strength of the penalty.
*   **Effect**: If a weight $w_j$ tries to grow large (e.g., 100), the penalty $100^2 = 10,000$ becomes huge. The model is forced to keep $w$ small.
""")

st.subheader("The Geometry: The Circle")
st.markdown(r"""
Imagine the Loss Function wants to go to the center (minimum error), but the Penalty holds it back like a leash.
For L2, the constraint region is a **Circle** ($w_1^2 + w_2^2 \le C$).
""")
st.markdown("`<img src='l2_circle.png'>`", unsafe_allow_html=True)

st.subheader("The Gradient Update")
st.markdown(r"""
When we take the derivative, an extra term appears:
""")
st.latex(r"\frac{\partial}{\partial w} (\lambda w^2) = 2\lambda w")
st.markdown(r"""
The update rule becomes:
""")
st.latex(r"w \leftarrow w - \eta (\nabla J + 2\lambda w)")
st.latex(r"w \leftarrow w(1 - 2\eta\lambda) - \eta \nabla J")
st.markdown(r"""
**Look closely!** Before updating based on the error, we multiply $w$ by $(1 - 2\eta\lambda)$.
Since this is $< 1$, we are **shrinking** the weight at every step. This is why L2 is called **Weight Decay**.
""")

# --- 3. L1 Regularization ---
st.header("3. L1 Regularization (Lasso) 💎")
st.markdown(r"""
**The Budget Cut.**

We add the **Sum of Absolute Weights** to the loss function.
""")
st.latex(r"J_{regularized}(w) = J(w) + \lambda \sum_{j=1}^n |w_j|")

st.subheader("The Geometry: The Diamond")
st.markdown(r"""
For L1, the constraint region is a **Diamond** ($|w_1| + |w_2| \le C$).
The "contours" of the Loss function often hit the **corners** of the diamond first.
At the corners, one of the weights is **exactly zero**.
""")
st.markdown("`<img src='l1_diamond.png'>`", unsafe_allow_html=True)

st.success("""
**Key Takeaway:** L1 Regularization can set weights to **Zero**. It performs **Feature Selection**.
It says: "This tennis stat is useless. Ignore it completely."
""")

# --- 4. Comparison ---
st.header("4. L1 vs L2: The Showdown 🥊")
st.markdown("""
| Feature | L2 (Ridge) | L1 (Lasso) |
| :--- | :--- | :--- |
| **Penalty** | Squared ($w^2$) | Absolute ($|w|$) |
| **Geometry** | Circle ⭕ | Diamond 💎 |
| **Effect** | Shrinks weights to near-zero | Sets weights to exactly zero |
| **Use Case** | Default choice. Prevents overfitting. | Feature Selection. Sparse models. |
""")

st.warning(r"""
**⚠️ The Scikit-Learn Trap**

In theory, we use $\lambda$ (Lambda) for penalty strength.
In `sklearn`, we use **`C`**.

$$C = \frac{1}{\lambda}$$

*   **High C** = Low Lambda = **Weak Regularization** (Trust the data).
*   **Low C** = High Lambda = **Strong Regularization** (Trust the prior/penalty).
""")

# --- 5. Interactive Playground ---
st.header("5. Interactive Playground 🎮")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Settings")
    reg_type = st.radio("Regularization Type", ["L2 (Ridge)", "L1 (Lasso)"])
    c_val = st.select_slider("C (Inverse Penalty)", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], value=1.0)
    noise = st.slider("Noise Level", 0.0, 1.0, 0.3)

    st.markdown("---")
    st.markdown("**Interpretation**")
    if c_val < 0.1:
        st.info("Strong Regularization. The model is very simple (maybe too simple).")
    elif c_val > 100:
        st.error("Weak Regularization. The model is overfitting (wiggly).")
    else:
        st.success("Balanced Regularization.")

with col2:
    # Generate Data
    X, y = make_moons(n_samples=200, noise=noise, random_state=42)

    # Pipeline: Polynomial Features -> Scaler -> LogReg
    # We need Polynomial features to show "wiggly" boundaries
    degree = 5

    if reg_type == "L1 (Lasso)":
        clf = LogisticRegression(C=c_val, penalty='l1', solver='liblinear', max_iter=1000)
    else:
        clf = LogisticRegression(C=c_val, penalty='l2', solver='lbfgs', max_iter=1000)

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        StandardScaler(),
        clf
    )

    model.fit(X, y)

    # Plotting
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_title(f"Decision Boundary (C={c_val}, {reg_type})")

    st.pyplot(fig)

    # Show Coefficients
    # Access the logistic regression step
    coefs = model.named_steps['logisticregression'].coef_.flatten()
    st.write(f"**Number of Features (Polynomial Degree {degree}):** {len(coefs)}")
    st.write(f"**Number of Non-Zero Weights:** {np.sum(np.abs(coefs) > 0.0001)}")

    if reg_type == "L1 (Lasso)":
        st.caption("Notice how L1 drives many weights to exactly zero!")

# --- 6. Summary & Quiz ---
st.header("6. Summary & Quiz 📝")
st.markdown("""
*   **Overfitting**: Memorizing noise.
*   **Regularization**: Adding a penalty to the loss function to keep weights small.
*   **L2 (Ridge)**: Squared penalty. Shrinks weights. Good default.
*   **L1 (Lasso)**: Absolute penalty. Zeroes out weights. Feature Selection.
*   **C**: Inverse of regularization strength. Low C = High Penalty.
""")

with st.expander("Quiz Question 1"):
    st.markdown("**Q: I have a dataset with 10,000 features, but I suspect only 20 are important. Which regularization should I use?**")
    st.markdown("**A:** **L1 (Lasso)**. It will drive the 9,980 useless features to zero, leaving you with the important ones.")

with st.expander("Quiz Question 2"):
    st.markdown("**Q: If I set C = 1,000,000, am I regularizing a lot or a little?**")
    st.markdown("**A:** **A little (or not at all)**. High C means Low Penalty. You are telling the model to trust the training data completely.")
