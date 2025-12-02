import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.toy_datasets import generate_moons

st.set_page_config(page_title="Hyperparameters", page_icon="🎛️", layout="wide")
sidebar_navigation()

st.title("🎛️ Hyperparameters & Grid Search")

# 1. Intuition
st.header("1. Intuition: Tuning the Radio")
st.markdown("""
### Parameters vs. Hyperparameters
A Machine Learning model has **two types of settings**:

1.  **Parameters ($w, b$)**: These are learned **automatically** by the model during training (using Gradient Descent).
    *   *Analogy*: The specific notes played by the music on the radio.
2.  **Hyperparameters**: These are settings **YOU** must choose *before* training starts. The model cannot learn them itself.
    *   *Analogy*: The **Volume Knob** and the **Station Tuner** on the radio.

If you are on the wrong station (wrong Hyperparameter), it doesn't matter how good the music is (Parameters); you will only hear static.

**Examples of Hyperparameters:**
*   **Learning Rate**: How fast to learn.
*   **Max Depth**: How deep the Decision Tree can grow.
*   **K**: Number of neighbors in KNN.
*   **C**: Regularization strength in SVM/Logistic Regression.
""")

st.markdown("---")

# 2. Math
st.header("2. The Math: Grid Search & Cross-Validation")
st.markdown("How do we find the best settings? We can't just guess. We use **Grid Search**.")
st.markdown("We define a 'grid' of possible values and try **every combination**.")

st.subheader("The Objective Function")
st.markdown("We want to find the set of hyperparameters $\\theta$ (theta) that minimizes the average validation error across $K$ folds.")

st.latex(r"\hat{\theta} = \arg \min_{\theta \in \Theta} \frac{1}{K}\sum_{k=1}^K L^{(k)}(\theta)")

st.markdown("""
*   $\Theta$: The Grid (e.g., C=[0.1, 1, 10], Gamma=[0.1, 1]).
*   $K$: The number of folds (usually 5).
*   $L^{(k)}(\theta)$: The Loss (Error) on the $k$-th validation fold using settings $\theta$.
""")

st.subheader("Why Cross-Validation?")
st.markdown("If we just test on one validation set, we might get lucky. Cross-Validation splits the data into $K$ parts. We train on $K-1$ and test on the last one, rotating $K$ times. This gives a much more robust estimate of performance.")

st.markdown("---")

# 3. Interactive Viz
st.header("3. Interactive Grid Search")
st.markdown("Let's tune an SVM on the 'Moons' dataset. We will vary **C** (Regularization) and **Gamma** (Kernel Coefficient).")

col1, col2 = st.columns([1, 2])

with col1:
    noise = st.slider("Dataset Noise", 0.0, 0.5, 0.3)

    st.markdown("### Grid Settings")
    c_values = [0.1, 1, 10, 100]
    gamma_values = [0.1, 1, 10, 100]

    st.write(f"Testing {len(c_values) * len(gamma_values)} combinations...")
    st.write("Using 3-Fold Cross-Validation.")

    if st.button("Run Grid Search"):
        X, y = generate_moons(noise=noise)

        results = []

        progress_bar = st.progress(0)
        total = len(c_values) * len(gamma_values)
        current = 0

        for c in c_values:
            for g in gamma_values:
                model = SVC(C=c, gamma=g, kernel='rbf')
                # 3-Fold CV
                scores = cross_val_score(model, X, y, cv=3)
                mean_score = scores.mean()

                results.append({'C': str(c), 'Gamma': str(g), 'Accuracy': mean_score})

                current += 1
                progress_bar.progress(current / total)

        results_df = pd.DataFrame(results)

        # Pivot for Heatmap
        heatmap_data = results_df.pivot(index='Gamma', columns='C', values='Accuracy')

        st.session_state['grid_results'] = heatmap_data

with col2:
    if 'grid_results' in st.session_state:
        st.subheader("Grid Search Results (Accuracy)")

        fig = px.imshow(
            st.session_state['grid_results'],
            labels=dict(x="C (Regularization)", y="Gamma (Kernel Coeff)", color="Accuracy"),
            x=st.session_state['grid_results'].columns,
            y=st.session_state['grid_results'].index,
            text_auto='.2f',
            color_continuous_scale='Viridis',
            origin='lower'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        best_acc = st.session_state['grid_results'].max().max()
        st.success(f"✅ Best Accuracy: {best_acc:.2%}")
        st.markdown("**Interpretation:** Brighter colors = Better performance. Look for the 'hot spot' on the grid.")
    else:
        st.info("👈 Click 'Run Grid Search' to start.")

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Back to Playground", icon="🎮")
