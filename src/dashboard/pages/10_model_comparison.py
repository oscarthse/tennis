import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.components.data_manager import get_train_test_data
from src.dashboard.components.model_manager import discover_models, load_model

st.set_page_config(page_title="Model Comparison", page_icon="⚔️", layout="wide")
sidebar_navigation()

st.title("⚔️ Model Comparison & Evaluation")

st.markdown("""
Here we evaluate all trained models on the **Test Set** (Matches from 2024).
This ensures we are measuring how well the models generalize to *future* data.
""")

# 1. Load Data
with st.spinner("Loading Test Data..."):
    X_train, X_test, y_train, y_test, test_df = get_train_test_data()

if X_test is None:
    st.error("Could not load test data. Please check the data path.")
    st.stop()

st.success(f"Loaded Test Set: **{len(X_test)} matches** (2024 Season)")

# 2. Load Models
models_info = discover_models()
if not models_info:
    st.warning("No models found. Please train models first.")
    st.stop()

results = []
roc_curves = []
calibration_curves = []

progress_bar = st.progress(0)
total_models = len(models_info)

for i, (name, info) in enumerate(models_info.items()):
    try:
        model = load_model(info['path'])

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": auc
        })

        # ROC Curve Data
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curves.append((name, fpr, tpr, auc))

            # Calibration Data
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
            calibration_curves.append((name, prob_true, prob_pred))

    except Exception as e:
        st.error(f"Error evaluating {name}: {e}")

    progress_bar.progress((i + 1) / total_models)

# 3. Display Results
st.markdown("### 📊 Performance Metrics")
results_df = pd.DataFrame(results).set_index("Model").sort_values("Accuracy", ascending=False)

st.dataframe(
    results_df.style.highlight_max(axis=0, color='lightgreen').format("{:.2%}"),
    use_container_width=True
)

# 4. Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))

    for name, fpr, tpr, auc_score in roc_curves:
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc_score:.2f})'))

    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500)
    st.plotly_chart(fig_roc, use_container_width=True)

with col2:
    st.subheader("Calibration Curves")
    st.markdown("*How well do predicted probabilities match actual win rates?*")
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Perfect'))

    for name, prob_true, prob_pred in calibration_curves:
        fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name=name))

    fig_cal.update_layout(xaxis_title="Predicted Probability", yaxis_title="Actual Win Rate", height=500)
    st.plotly_chart(fig_cal, use_container_width=True)

st.markdown("---")
st.page_link("pages/02_model_playground.py", label="🎮 Go to Playground", icon="🎮")
