import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.math_blocks import get_metrics_latex

st.set_page_config(page_title="Metrics & Validation", page_icon="✅", layout="wide")
sidebar_navigation()

st.title("✅ Metrics & Validation: Beyond Accuracy")

# 1. Intuition
st.header("1. Intuition: The Accuracy Paradox")
st.markdown("""
### The Problem with Accuracy
Imagine you are building a **Cancer Detector**.
*   **990** patients are Healthy.
*   **10** patients have Cancer.

You build a lazy model that just says **"Healthy" for everyone**.
*   It is correct 990 times out of 1000.
*   **Accuracy = 99%**.

**Is this a good model?**
**NO!** It killed 10 people. It failed to find the *only thing we cared about*.

This is why we need **Precision** and **Recall**.
""")

st.markdown("---")

# 2. Math
st.header("2. The Math: The Confusion Matrix")
st.markdown("We break down predictions into 4 buckets:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    | | Predicted Negative (0) | Predicted Positive (1) |
    |---|---|---|
    | **Actual Negative (0)** | **True Negative (TN)**<br>*(Correctly ignored)* | **False Positive (FP)**<br>*(False Alarm)* |
    | **Actual Positive (1)** | **False Negative (FN)**<br>*(Missed Opportunity)* | **True Positive (TP)**<br>*(Success!)* |
    """)

with col2:
    math = get_metrics_latex()

    st.subheader("Precision")
    st.latex(math["precision"])
    st.markdown("*\"When we claimed it was Cancer, how often were we right?\"*")
    st.markdown("High Precision = Few False Alarms.")

    st.subheader("Recall (Sensitivity)")
    st.latex(math["recall"])
    st.markdown("*\"Of all the actual Cancer cases, how many did we find?\"*")
    st.markdown("High Recall = Few Missed Cases.")

    st.subheader("F1 Score")
    st.latex(math["f1"])
    st.markdown("The harmonic mean. Good for balancing Precision and Recall.")

st.markdown("---")

# 3. Worked Example
st.header("3. Worked Example")
st.markdown("Let's calculate metrics for a Tennis Model.")

st.markdown("""
*   **TP (Predicted Win, Actual Win)**: 30
*   **FP (Predicted Win, Actual Loss)**: 20
*   **FN (Predicted Loss, Actual Win)**: 10
*   **TN (Predicted Loss, Actual Loss)**: 40
""")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### Precision")
    st.latex(r"\frac{30}{30 + 20} = \frac{30}{50} = 0.60")
    st.markdown("60% of our 'Win' predictions were correct.")

with c2:
    st.markdown("#### Recall")
    st.latex(r"\frac{30}{30 + 10} = \frac{30}{40} = 0.75")
    st.markdown("We correctly identified 75% of the actual wins.")

with c3:
    st.markdown("#### F1 Score")
    st.latex(r"2 \cdot \frac{0.6 \cdot 0.75}{0.6 + 0.75} = 2 \cdot \frac{0.45}{1.35} \approx 0.67")

st.markdown("---")

# 4. Interactive Viz
st.header("4. Interactive Visualization")
st.markdown("Adjust the **Decision Threshold** to see how it affects the Confusion Matrix and Metrics.")
st.markdown("*   **Low Threshold** (e.g., 0.1): Predict 'Win' even if probability is low. Increases Recall, lowers Precision.")
st.markdown("*   **High Threshold** (e.g., 0.9): Predict 'Win' only if super confident. Increases Precision, lowers Recall.")

col1, col2 = st.columns([1, 2])

with col1:
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
    st.info("Lower threshold = More Positive Predictions.")

with col2:
    # Generate Synthetic Probabilities
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    # Generate probabilities correlated with y_true but noisy
    y_prob = np.random.rand(n_samples)
    y_prob = y_prob * 0.4 + y_true * 0.6 # Shift distribution based on class

    # Apply Threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate Metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / n_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Display Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.2%}")
    m2.metric("Precision", f"{precision:.2%}")
    m3.metric("Recall", f"{recall:.2%}")
    m4.metric("F1 Score", f"{f1:.2f}")

    # Plot Confusion Matrix
    fig = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=['Pred Negative', 'Pred Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',
        text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    fig.update_layout(title="Confusion Matrix", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))

    # Current Point
    current_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    current_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fig_roc.add_trace(go.Scatter(x=[current_fpr], y=[current_tpr], mode='markers', marker=dict(color='red', size=12), name='Current Threshold'))

    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
    st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Back to Playground", icon="🎮")
