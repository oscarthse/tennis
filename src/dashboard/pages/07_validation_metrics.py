import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
from src.dashboard.components.navigation import sidebar_navigation

st.set_page_config(page_title="Validation & Metrics", page_icon="✅", layout="wide")
sidebar_navigation()

st.title("✅ Validation & Metrics: Beyond Accuracy")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown(r"""
**Accuracy is a Lie.**
If 99% of transactions are legit, a model that says "Legit" 100% of the time has 99% Accuracy but **Zero Intelligence**.

We need granular metrics derived from the **Confusion Matrix**:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**The Truth**")
    st.markdown("*   **Positive (1)**: Win / Fraud / Sick")
    st.markdown("*   **Negative (0)**: Lose / Legit / Healthy")
with col2:
    st.markdown("**The Prediction**")
    st.markdown("*   **TP**: Hit (Correct)")
    st.markdown("*   **FP**: False Alarm (Type I)")
    st.markdown("*   **FN**: Miss (Type II)")
    st.markdown("*   **TN**: Correct Rejection")

# --- 2. The Tradeoff (Platinum Depth) ---
st.header("2. The Precision-Recall Tradeoff")
st.markdown(r"""
You cannot have it all. You must choose what you care about.

*   **Precision**: $\frac{TP}{TP + FP}$. "When I say it's a Win, am I lying?"
    *   **High Precision Needed**: Spam Filter (Don't delete important emails), Stock Picking (Don't lose money).
*   **Recall**: $\frac{TP}{TP + FN}$. "Did I find all the Wins?"
    *   **High Recall Needed**: Cancer Detection (Don't miss a sick patient), Fraud Detection (Catch all thieves).

**The Threshold ($T$):**
*   $P(\text{Win}) > T \implies \text{Predict Win}$.
*   **Raise $T$** (e.g., 0.9): Fewer predictions, but higher confidence. **Precision $\uparrow$, Recall $\downarrow$**.
*   **Lower $T$** (e.g., 0.1): More predictions, catch everything. **Recall $\uparrow$, Precision $\downarrow$**.
""")

# --- 3. Advanced Metrics (Platinum Depth) ---
st.header("3. Advanced Metrics")

tab_auc, tab_f1, tab_mcc = st.tabs(["AUC-ROC", "F1 Score", "Kappa & MCC"])

with tab_auc:
    st.subheader("AUC-ROC: The Probabilistic View")
    st.markdown(r"""
    **ROC Curve**: Plot of TPR (Recall) vs FPR (False Alarm Rate) at *all possible thresholds*.
    **AUC (Area Under Curve)**: A single number summary (0.5 = Random, 1.0 = Perfect).

    **Deep Intuition**:
    AUC is the probability that the model ranks a random **Positive** example higher than a random **Negative** example.
    """)
    st.latex(r"AUC = P(\text{Score}(x_{pos}) > \text{Score}(x_{neg}))")

with tab_f1:
    st.subheader("F1 Score: The Harmonic Mean")
    st.markdown(r"""
    If Precision=0.01 and Recall=1.0 (The "Predict All" strategy), the Arithmetic Mean is 0.5 (Misleading).
    The **Harmonic Mean** punishes extreme values.
    """)
    st.latex(r"F1 = 2 \cdot \frac{P \cdot R}{P + R}")
    st.markdown("If either P or R is near 0, F1 crashes to 0.")

with tab_mcc:
    st.subheader("Matthews Correlation Coefficient (MCC)")
    st.markdown(r"""
    The **Gold Standard** for binary classification, especially with imbalance.
    It treats all 4 quadrants of the confusion matrix equally.
    Range: [-1, +1].
    *   +1: Perfect.
    *   0: Random guessing.
    *   -1: Perfectly wrong (Inverse prediction).
    """)

# --- 6. Visualization ---
st.header("6. Visualization: The Threshold Slider")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5)
    noise = st.slider("Noise", 0.1, 2.0, 1.0)
    separation = st.slider("Separation", 0.0, 5.0, 2.0)

with col_viz:
    # Generate Data
    n = 1000
    # Class 0: N(0, 1)
    # Class 1: N(separation, 1)
    neg = np.random.normal(0, 1, n)
    pos = np.random.normal(separation, 1, n)

    y_true = np.concatenate([np.zeros(n), np.ones(n)])
    y_scores = np.concatenate([neg, pos])

    # Metrics at current threshold
    y_pred = (y_scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # Plots
    fig = go.Figure()

    # Distributions
    fig.add_trace(go.Histogram(x=neg, name='Class 0 (Neg)', opacity=0.5, marker_color='red', nbinsx=50))
    fig.add_trace(go.Histogram(x=pos, name='Class 1 (Pos)', opacity=0.5, marker_color='blue', nbinsx=50))

    # Threshold Line
    fig.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="black", annotation_text="Threshold")

    fig.update_layout(barmode='overlay', title="Score Distributions", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Metrics Display
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{prec:.2f}")
    c2.metric("Recall", f"{rec:.2f}")
    c3.metric("F1 Score", f"{f1:.2f}")
    c4.metric("Accuracy", f"{(tp+tn)/(2*n):.2f}")

# --- 8. Super Summary ---
st.header("8. Super Summary 🦸")
st.info(r"""
*   **Precision**: Quality of positive predictions.
*   **Recall**: Quantity of positive predictions found.
*   **Threshold**: The knob that trades P for R.
*   **AUC**: How well separated the distributions are (independent of threshold).
""")
