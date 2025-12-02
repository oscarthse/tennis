import streamlit as st
import pandas as pd
from src.dashboard.utils.data_manager import get_train_test_data
from src.dashboard.utils.model_manager import load_model
from src.dashboard.utils.metrics import compute_metrics, get_confusion_matrix
from src.dashboard.utils.plotting import plot_confusion_matrix, plot_calibration_curve, plot_grouped_accuracy

def render_evaluation(available_models):
    """Render the Evaluation tab."""
    st.header('🔬 Model Evaluation Explorer')

    if not available_models:
        st.warning("No models found.")
        return

    col_sel, _ = st.columns([1, 3])
    with col_sel:
        selected_model_name = st.selectbox('Select Model to Evaluate', list(available_models.keys()))

    model_info = available_models[selected_model_name]

    # Load Data
    _, X_test, _, y_test, df_test = get_train_test_data()

    if X_test is None:
        st.error("Could not load test data.")
        return

    # Load Model & Predict
    try:
        model = load_model(model_info['path'])
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    except Exception as e:
        st.error(f"Error loading/predicting with model: {e}")
        return

    # Overall Metrics
    st.subheader('📊 Overall Performance (Test Set)')
    metrics = compute_metrics(y_test, y_pred, y_prob)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}" if metrics['ROC-AUC'] else "N/A")
    c3.metric("Brier Score", f"{metrics['Brier Score']:.4f}")
    c4.metric("Log Loss", f"{metrics['Log Loss']:.4f}" if metrics['Log Loss'] else "N/A")

    st.markdown('---')

    # Plots
    col1, col2 = st.columns(2)

    with col1:
        cm = get_confusion_matrix(y_test, y_pred)
        fig_cm = plot_confusion_matrix(cm)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        if hasattr(model, 'predict_proba'):
            fig_cal = plot_calibration_curve(y_test, y_prob)
            st.plotly_chart(fig_cal, use_container_width=True)
        else:
            st.info("Calibration curve requires predict_proba")

    st.markdown('---')

    # Sliced Analysis
    st.subheader('🔍 Sliced Analysis')

    # Prepare analysis dataframe
    df_analysis = df_test.copy()
    df_analysis['Actual'] = y_test
    df_analysis['Predicted'] = y_pred
    df_analysis['Correct'] = (df_analysis['Actual'] == df_analysis['Predicted'])

    slice_col1, slice_col2 = st.columns(2)

    with slice_col1:
        # Surface Accuracy
        surface_acc = df_analysis.groupby('Surface')['Correct'].mean().reset_index(name='Accuracy')
        fig_surf = plot_grouped_accuracy(surface_acc, 'Surface', 'Accuracy by Surface')
        st.plotly_chart(fig_surf, use_container_width=True)

    with slice_col2:
        # Round Accuracy
        round_acc = df_analysis.groupby('Round')['Correct'].mean().reset_index(name='Accuracy').sort_values('Accuracy', ascending=False)
        fig_round = plot_grouped_accuracy(round_acc, 'Round', 'Accuracy by Round')
        st.plotly_chart(fig_round, use_container_width=True)
