import streamlit as st
from src.dashboard.config import PAGE_ICON

def render_home(df):
    """Render the Home tab."""
    st.header(f'{PAGE_ICON} Welcome to Tennis ML Predictor')

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Matches', f'{len(df):,}')
    with col2:
        st.metric('Players', f'{df["Player_1"].nunique() + df["Player_2"].nunique():,}')
    with col3:
        st.metric('Tournaments', f'{df["Tournament"].nunique():,}')

    st.markdown('---')

    # Instructions
    st.subheader('📖 How to Use')
    st.markdown("""
    This dashboard allows you to explore machine learning models trained on ATP tennis match data.

    1. **🎮 Playground**:
       - Select two players and match conditions.
       - The system auto-fills player stats (Rank, Points) based on the latest data.
       - Predict the winner and see confidence scores.

    2. **📊 Comparison**:
       - Compare multiple trained models side-by-side.
       - View metrics like Accuracy, ROC-AUC, and Log Loss on the test set.

    3. **🔬 Evaluation**:
       - Deep dive into a specific model's performance.
       - Analyze confusion matrices, calibration curves, and performance by surface.
    """)

    # Dataset Preview
    with st.expander("📋 View Dataset Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True, height=300)
