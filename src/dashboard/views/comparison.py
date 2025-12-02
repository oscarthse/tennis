import streamlit as st
import pandas as pd
from src.dashboard.utils.data_manager import get_train_test_data
from src.dashboard.utils.model_manager import load_model
from src.dashboard.utils.metrics import compute_metrics
from src.dashboard.utils.plotting import plot_metric_comparison

def render_comparison(available_models):
    """Render the Comparison tab."""
    st.header('📊 Model Comparison')

    if len(available_models) < 1:
        st.info('Train models to enable comparison.')
        return

    # Load Test Data
    _, X_test, _, y_test, _ = get_train_test_data()

    if X_test is None:
        st.error("Could not load test data.")
        return

    st.markdown(f"Evaluating on **{len(X_test)}** test samples.")

    # Calculate Metrics
    results = []
    progress_bar = st.progress(0)

    for i, (name, info) in enumerate(available_models.items()):
        try:
            model = load_model(info['path'])
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics['Model'] = name
            results.append(metrics)
        except Exception as e:
            st.error(f"Error evaluating {name}: {e}")

        progress_bar.progress((i + 1) / len(available_models))

    progress_bar.empty()

    if not results:
        return

    df_results = pd.DataFrame(results)

    # Metrics Table
    st.subheader('📈 Performance Metrics')
    st.dataframe(
        df_results.set_index('Model').style.highlight_max(axis=0, color='#d4edda').format("{:.4f}"),
        use_container_width=True
    )

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        fig_acc = plot_metric_comparison(df_results, 'Accuracy', 'Accuracy Comparison', 'Blues')
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        fig_auc = plot_metric_comparison(df_results, 'ROC-AUC', 'ROC-AUC Comparison', 'Greens')
        st.plotly_chart(fig_auc, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
         fig_brier = plot_metric_comparison(df_results, 'Brier Score', 'Brier Score (Lower is Better)', 'Reds_r')
         st.plotly_chart(fig_brier, use_container_width=True)
