import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.toy_datasets import generate_moons

st.set_page_config(page_title="Validation & Metrics", page_icon="✅", layout="wide")
sidebar_navigation()

st.title("✅ Validation & Metrics")

# --- 1. Core Model Definition ---
st.header("1. Core Model Definition")
st.markdown("""
Accuracy is not enough. We need to understand **how** the model is wrong.
The foundation of all metrics is the **Confusion Matrix**.

**The Four Outcomes:**
*   **True Positive (TP)**: Predicted Win, Actually Win. (Correct)
*   **False Positive (FP)**: Predicted Win, Actually Lose. (Type I Error / False Alarm).
*   **False Negative (FN)**: Predicted Lose, Actually Win. (Type II Error / Miss).
*   **True Negative (TN)**: Predicted Lose, Actually Lose. (Correct).
""")

# --- 2. Geometry / Structure ---
st.header("2. Geometry: The Threshold Slider")
st.markdown("""
Most models output a probability (e.g., 0.7). We need a **Threshold** (usually 0.5) to make a decision.
*   If $p > T$: Predict Positive.
*   If $p < T$: Predict Negative.

Moving this threshold $T$ trades off FPs and FNs.
*   **Low Threshold (0.1)**: Predict "Win" aggressively. High Recall, Low Precision. (Catch all wins, but many false alarms).
*   **High Threshold (0.9)**: Predict "Win" conservatively. High Precision, Low Recall. (Only bet on sure things).
""")

# --- 3. Constraints / Objective / Loss ---
st.header("3. The Metrics")
st.markdown("We define derived metrics to capture specific behaviors.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Precision")
    st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
    st.markdown("**Interpretation**: When the model says 'Win', how often is it right? (Trustworthiness).")

with col2:
    st.subheader("Recall (Sensitivity)")
    st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
    st.markdown("**Interpretation**: Out of all actual Wins, how many did we find? (Coverage).")

st.subheader("F1 Score")
st.latex(r"F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
st.markdown("The Harmonic Mean. It punishes extreme values (e.g., if Precision is 0, F1 is 0).")

# --- 6. Visualization ---
st.header("6. Visualization: ROC & PR Curves")

col_viz, col_controls = st.columns([3, 1])
with col_controls:
    noise = st.slider("Noise", 0.1, 1.0, 0.5)
    imbalance = st.slider("Imbalance (Ratio 0:1)", 0.1, 0.9, 0.5)

with col_viz:
    # Generate Data
    n_samples = 500
    n_class1 = int(n_samples * imbalance)
    n_class0 = n_samples - n_class1

    # Simple synthetic scores
    # Class 0: N(0, 1), Class 1: N(1 + noise, 1)
    scores_0 = np.random.normal(0, 1, n_class0)
    scores_1 = np.random.normal(2 - noise, 1, n_class1)

    y_true = np.concatenate([np.zeros(n_class0), np.ones(n_class1)])
    y_scores = np.concatenate([scores_0, scores_1])

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate (1-Specificity)", yaxis_title="True Positive Rate (Recall)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Top-Left is best (High TPR, Low FPR).")

# --- 8. Super Summary ---
st.header("8. Super Summary 🦸")
st.info("""
*   **Goal**: Evaluate model performance beyond simple accuracy.
*   **Math**: Precision ($TP/PredP$), Recall ($TP/TrueP$).
*   **Key Insight**: There is always a trade-off between Precision and Recall. You choose it by setting the Threshold.
*   **Knobs**: The Decision Threshold.
""")
