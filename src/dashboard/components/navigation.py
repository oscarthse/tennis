import streamlit as st

def sidebar_navigation():
    """
    Render the sidebar navigation.
    Includes error handling for standalone page execution.
    """
    def safe_link(page, label, icon):
        try:
            st.sidebar.page_link(page, label=label, icon=icon)
        except KeyError:
            # Fallback for standalone execution where pages aren't registered
            st.sidebar.markdown(f"{icon} **{label}**")
        except Exception:
            st.sidebar.markdown(f"{icon} {label}")

    st.sidebar.header("🎾 Tennis ML Platform")

    st.sidebar.markdown("### 📚 Syllabus")
    safe_link("app.py", label="Home / Overview", icon="🏠")
    safe_link("pages/01_data_and_features.py", label="1. Data & Features", icon="📊")

    st.sidebar.markdown("### 🤖 Models")
    safe_link("pages/03_learn_log_reg.py", label="2. Logistic Regression", icon="📈")
    safe_link("pages/04_learn_trees_forests.py", label="3. Trees & Forests", icon="🌳")
    safe_link("pages/05_learn_svm_knn.py", label="4. SVM & KNN", icon="📐")

    st.sidebar.markdown("### 🛠️ Optimization & Eval")
    safe_link("pages/06_optimization.py", label="5. Gradient Descent", icon="📉")
    safe_link("pages/07_validation_metrics.py", label="6. Metrics & Validation", icon="✅")
    safe_link("pages/08_hyperparameters.py", label="7. Hyperparameters", icon="🎛️")
    safe_link("pages/08b_regularization.py", label="7b. Regularization (L1/L2)", icon="🛡️")
    safe_link("pages/09_class_imbalance.py", label="8. Class Imbalance", icon="⚖️")
    safe_link("pages/10_model_comparison.py", label="9. Model Comparison", icon="⚔️")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧩 Software Dev / DevOps")
    safe_link("pages/11_computer_architecture.py", label="1. Computer Architecture", icon="💻")
    safe_link("pages/12_docker_basics.py", label="2. Docker Basics", icon="🐳")
    safe_link("pages/13_docker_networking.py", label="3. Docker Networking", icon="🌐")
    safe_link("pages/14_docker_compose.py", label="4. Docker Compose", icon="📦")
    safe_link("pages/18_docker_advanced.py", label="5. Docker Advanced", icon="🚀")
    safe_link("pages/15_git_basics.py", label="Git Basics", icon="🌿")
    safe_link("pages/16_git_collaboration.py", label="Git Collaboration", icon="🤝")
    safe_link("pages/17_git_advanced.py", label="Git Advanced", icon="🧠")

    st.sidebar.markdown("---")
    safe_link("pages/02_model_playground.py", label="🎮 Model Playground", icon="🎮")
    st.sidebar.info("Use the Playground to test models on real tennis data!")
