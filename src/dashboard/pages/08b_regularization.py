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

st.set_page_config(page_title="L1 & L2 Regularization", page_icon="🛡️", layout="wide")
sidebar_navigation()

st.title("🛡️ L1 (LASSO) and L2 (Ridge) Regularization")
st.subheader("Mathematical Foundations and MLOps Verification")

st.markdown("""
**Elevator Pitch:** Regularization is the mathematical mechanism for controlling the **Bias-Variance Tradeoff**.
By imposing a structured penalty on the model's complexity (weights), we prevent overfitting and ensure generalization to unseen data.
""")

st.markdown("### Learning Outcomes")
st.markdown(r"""
*   **Derivation**: Derive the exact gradient update rules for L2 (Weight Decay) and L1 (Soft Thresholding).
*   **Geometry**: Visualize the optimization landscape using **Lagrange Multipliers** (Circle vs. Diamond).
*   **Bias-Variance**: Deeply understand how $\lambda$ shifts the model from High Variance to High Bias.
*   **Verification**: Learn how to write **Unit Tests** to mathematically verify that your regularization is functioning correctly in production.
""")

# --- 2. Motivation: Bias-Variance Deep Dive ---
st.header("1. Motivation: The \"Memorizing\" Player (Bias-Variance Deep Dive) 🧠")
st.markdown(r"""
Let's revisit our tennis analogy with more rigor. We are trying to learn a function $f(x)$ that maps match conditions to the opponent's shot.

### The Tradeoff
*   **Bias (Underfitting)**: The model is too simple. It assumes the opponent *always* hits cross-court, ignoring all other signals. It fails to capture the true underlying pattern.
*   **Variance (Overfitting)**: The model is too complex. It learns a unique rule for every single point in history.
    *   *Example*: "If opponent wears a green shirt AND wind is 5mph $\to$ Drop Shot."
    *   This is **High Variance** because the model's decision boundary changes wildly with small fluctuations in the training data (noise).

### The Unregularized Objective
In standard Logistic Regression, we minimize the negative log-likelihood (Log Loss):

$$J(w, b) = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

**The Danger**: If we minimize *only* this term, and we have high-dimensional features (e.g., polynomial expansion), the optimizer can drive $\|w\| \to \infty$ to fit every outlier perfectly.
""")

# --- 3. L2 Regularization (Ridge) ---
st.header("2. L2 Regularization (Ridge): The Weight Decay Mechanism ⭕")
st.markdown(r"""
**Formal Definition**: We add a penalty proportional to the **squared Euclidean norm** of the weights.

$$J_{\text{L2}}(w) = J(w) + \lambda \|w\|_2^2 = J(w) + \lambda \sum_{j=1}^d w_j^2$$

*   **$\lambda$ (Lambda)**: The regularization strength (Hyperparameter).
*   **Intuition**: This acts like a "Power Limiter". It allows the model to use weights to reduce error, but "taxes" large weights heavily.
""")

st.subheader("The Complete Derivation (Step-by-Step)")
st.markdown(r"""
Let's look at the Gradient Descent update.
Recall the gradient of the unregularized loss: $\nabla_w J$.

**Step 1: Gradient of the Penalty**
$$\frac{\partial}{\partial w_j} (\lambda \sum w_k^2) = 2\lambda w_j$$

**Step 2: The Full Update Rule**
$$w_{new} \leftarrow w_{old} - \eta (\nabla J + 2\lambda w_{old})$$

**Step 3: Rearranging for Insight (Weight Decay)**
$$w_{new} \leftarrow w_{old} - \eta \nabla J - 2\eta\lambda w_{old}$$
$$w_{new} \leftarrow \underbrace{(1 - 2\eta\lambda)}_{\text{Shrinkage Factor}} w_{old} - \eta \nabla J$$

**Key Insight**: The term $(1 - 2\eta\lambda)$ is slightly less than 1 (e.g., 0.99).
In *every single step*, before the model learns from the data ($\nabla J$), it **shrinks** the existing weights towards zero.
This is why L2 is mathematically identical to **Weight Decay**.
""")

st.subheader("Geometric Rigor: Lagrange Multipliers")
st.markdown(r"""
We can view this as a constrained optimization problem:
**Minimize $J(w)$ subject to $\sum w_j^2 \le t$**.

The optimal solution occurs where the **contours of the Loss Function** are **tangent** to the **Constraint Region**.
For L2, the constraint $\sum w_j^2 \le t$ defines a **Circle** (or Hypersphere).
""")
st.markdown("`<img src='l2_circle.png'>`", unsafe_allow_html=True)
st.caption("The elliptical loss contours touch the circular constraint. This rarely happens exactly at an axis, so weights are small but non-zero.")

# --- 4. L1 Regularization (Lasso) ---
st.header("3. L1 Regularization (Lasso): The Feature Selector 💎")
st.markdown(r"""
**Formal Definition**: We add a penalty proportional to the **L1 norm** (sum of absolute values).

$$J_{\text{L1}}(w) = J(w) + \lambda \|w\|_1 = J(w) + \lambda \sum_{j=1}^d |w_j|$$

**Intuition**: The "Minimalist Budget". It forces the model to spend its "weight budget" only on the most critical features.
""")

st.subheader("The Gradient and Sparsity")
st.markdown(r"""
The derivative of $|w|$ is not defined at 0. We use the **Subgradient**:
$$ \text{sign}(w_j) = \begin{cases} 1 & w_j > 0 \\ -1 & w_j < 0 \\ [-1, 1] & w_j = 0 \end{cases} $$

The gradient of the penalty is constant: $\pm \lambda$.
Unlike L2, where the push gets smaller as $w \to 0$ (since $2\lambda w \to 0$), L1 pushes with **constant force** $\lambda$ all the way to zero.
This leads to the **Soft Thresholding** effect, where weights effectively "snap" to zero.
""")

st.subheader("Geometric Rigor: The Diamond")
st.markdown(r"""
Constrained Optimization: **Minimize $J(w)$ subject to $\sum |w_j| \le t$**.
The constraint $\sum |w_j| \le t$ defines a **Diamond** (or L1 Ball).
""")
st.markdown("`<img src='l1_diamond.png'>`", unsafe_allow_html=True)
st.caption("The loss contours often hit the **corners** of the diamond first. At a corner, one or more coordinates are **exactly zero**. This is the geometric origin of **Sparsity**.")

# --- 5. Practical Implementation & MLOps ---
st.header("4. Practical Implementation & MLOps Verification 🛠️")

st.markdown("""
| Feature | L2 (Ridge) | L1 (Lasso) |
| :--- | :--- | :--- |
| **Penalty** | Squared ($w^2$) | Absolute ($|w|$) |
| **Shrinkage** | Proportional to weight | Constant force |
| **Result** | Small, dense weights | Sparse weights (Zeros) |
| **Use Case** | Default for performance | Feature Selection / Interpretability |
""")

st.subheader("Code Quality & Verification")
st.markdown("""
In a professional MLOps environment, we do not trust "theoretical" regularization. We verify it with automated tests.
""")

st.markdown("**1. Unit Testing (Pytest)**")
st.markdown("We write tests to assert the mathematical properties of our model.")
st.code("""
# tests/test_regularization.py (Excerpt)

def test_regularization_shrinks_weights():
    # Train High C (Weak Reg) vs Low C (Strong Reg)
    # Assert that Strong Reg norm is < 50% of Weak Reg norm
    assert reduction_ratio < 0.5

def test_l1_sparsity():
    # Train L1 vs L2
    # Assert L1 has more zero coefficients
    assert l1_zeros > l2_zeros
""", language="python")

st.markdown("**2. CI/CD Integration**")
st.markdown("""
These tests should run automatically in your **CI/CD Pipeline** (e.g., GitHub Actions).
*   **Trigger**: On every `git push`.
*   **Action**: Run `pytest`.
*   **Blocker**: If `test_regularization_shrinks_weights` fails, the code is rejected. This prevents you from accidentally deploying a model where regularization is broken (e.g., by passing the wrong argument).
""")

# --- Interactive Playground ---
st.header("5. Interactive Verification 🎮")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Settings")
    reg_type = st.radio("Regularization Type", ["L2 (Ridge)", "L1 (Lasso)"])
    c_val = st.select_slider("C (Inverse Penalty)", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], value=1.0)
    noise = st.slider("Noise Level", 0.0, 1.0, 0.3)

    st.markdown("---")
    if c_val < 0.1:
        st.info("High Bias Zone (Underfitting)")
    elif c_val > 100:
        st.error("High Variance Zone (Overfitting)")
    else:
        st.success("Balanced Zone")

with col2:
    X, y = make_moons(n_samples=200, noise=noise, random_state=42)

    if reg_type == "L1 (Lasso)":
        clf = LogisticRegression(C=c_val, penalty='l1', solver='liblinear', max_iter=1000)
    else:
        clf = LogisticRegression(C=c_val, penalty='l2', solver='lbfgs', max_iter=1000)

    model = make_pipeline(PolynomialFeatures(degree=5), StandardScaler(), clf)
    model.fit(X, y)

    # Plotting
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_title(f"Decision Boundary (C={c_val}, {reg_type})")
    st.pyplot(fig)

    coefs = model.named_steps['logisticregression'].coef_.flatten()
    st.write(f"**Non-Zero Weights:** {np.sum(np.abs(coefs) > 0.0001)} / {len(coefs)}")

# --- 6. Summary & Assessment ---
st.header("6. Summary & Assessment 📝")
st.markdown(r"""
*   **L2 (Ridge)**: Adds $\lambda \sum w^2$. Shrinks weights via $(1-2\eta\lambda)$ factor. Use for preventing overfitting while keeping all features.
*   **L1 (Lasso)**: Adds $\lambda \sum |w|$. Snaps weights to zero via constant gradient. Use for feature selection.
*   **MLOps**: Verification is key. Use unit tests to prove your regularization is active.
""")

with st.expander("Mastery Question 1"):
    st.markdown(r"**Q: Derive the shrinkage factor for L2 if $\eta=0.01$ and $\lambda=1.5$.**")
    st.markdown(r"**A:** The factor is $(1 - 2\eta\lambda) = 1 - 2(0.01)(1.5) = 1 - 0.03 = \mathbf{0.97}$. The weights shrink by 3% per step.")

with st.expander("Mastery Question 2"):
    st.markdown("**Q: Why does L1 lead to sparsity while L2 does not?**")
    st.markdown(r"**A:** Geometrically, the L1 'Diamond' has corners where the solution is likely to land. Mathematically, the L1 gradient is constant ($\pm \lambda$) even as $w \to 0$, pushing it all the way to zero, whereas the L2 gradient ($2\lambda w$) vanishes as $w \to 0$.")
