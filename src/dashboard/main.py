import streamlit as st
from src.dashboard.config import PAGE_TITLE, PAGE_ICON, LAYOUT
from src.dashboard.utils.data_manager import load_data
from src.dashboard.utils.model_manager import discover_models
from src.dashboard.views.home import render_home
from src.dashboard.views.playground import render_playground
from src.dashboard.views.comparison import render_comparison
from src.dashboard.views.evaluation import render_evaluation

# Page Config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; font-weight: 600;}
    h2 {color: #34495e; font-weight: 500;}
    h3 {color: #7f8c8d;}
    .stMetric {background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background-color: white; border-radius: 8px 8px 0 0; padding: 10px 20px;}
    div[data-testid="stExpander"] {background-color: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

def main():
    # Load Data
    df = load_data()
    if df.empty:
        st.error("Failed to load data. Please check the data file.")
        return

    # Discover Models
    available_models = discover_models()

    # Sidebar
    with st.sidebar:
        st.header(f"{PAGE_ICON} Navigation")
        st.info(f"Loaded **{len(available_models)}** models.")

        if st.button("🔄 Refresh Models"):
            discover_models.clear()
            st.rerun()


        if available_models:
            st.markdown("### Available Models")
            for name in available_models:
                st.text(f"• {name}")

        st.markdown("---")
        st.caption("v2.0.0 | Tennis ML Project")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(['🏠 Home', '🎮 Playground', '📊 Comparison', '🔬 Evaluation'])

    with tab1:
        render_home(df)

    with tab2:
        render_playground(df, available_models)

    with tab3:
        render_comparison(available_models)

    with tab4:
        render_evaluation(available_models)

if __name__ == '__main__':
    main()
