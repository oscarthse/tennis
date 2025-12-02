import streamlit as st
from src.dashboard.components.navigation import sidebar_navigation
from src.dashboard.config import DATA_PATH
import pandas as pd

st.set_page_config(
    page_title="Tennis ML Platform",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

sidebar_navigation()

st.title("🎾 Tennis Match Prediction: From Intuition to Production")

st.markdown("""
### Welcome to the Interactive Machine Learning Course!

This platform is designed to take you from **zero to hero** in understanding how we predict tennis match outcomes using Machine Learning.

We won't just show you the code. We will:
1.  **Build Intuition**: Understand *why* a model works using simple analogies.
2.  **Derive the Math**: See the actual equations behind the magic (Logistic Loss, Entropy, Gradients).
3.  **Write the Code**: See the clean Python implementation.
4.  **Play with Data**: Test the models on **real ATP Tennis data** in our Playground.

---

### 📚 The Syllabus

#### **Part 1: The Basics**
*   **[1. Data & Features](pages/01_data_and_features.py)**: What data do we have? What is "Elo"? How do we measure "Form"?
*   **[2. Logistic Regression](pages/03_learn_log_reg.py)**: The foundation of classification. Probabilities, Sigmoids, and the Log-Loss.

#### **Part 2: Non-Linear Models**
*   **[3. Trees & Forests](pages/04_learn_trees_forests.py)**: How to make decisions like a human. Entropy, Information Gain, and Bagging.
*   **[4. SVM & KNN](pages/05_learn_svm_knn.py)**: Geometry-based classification. Margins and Neighbors.

#### **Part 3: Optimization & Evaluation**
*   **[5. Gradient Descent](pages/06_optimization.py)**: How models actually "learn" by sliding down a hill.
*   **[6. Metrics & Validation](pages/07_validation_metrics.py)**: Accuracy is not enough! Precision, Recall, ROC Curves, and Confusion Matrices.
*   **[7. Hyperparameters](pages/08_hyperparameters.py)**: Tuning the knobs of your machine. Grid Search and Cross-Validation.
*   **[8. Class Imbalance](pages/09_class_imbalance.py)**: What happens when one player wins 90% of the time?

---

### 🚀 Quick Start
Want to jump straight to the action?
Go to the **[Model Playground](pages/02_model_playground.py)** to test our trained models (`XGBoost_v2`, `RandomForest_v2`) on the latest matches.

""")

# Quick Stats
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.info(f"📊 **Dataset Stats**: Loaded **{len(df):,}** matches from the ATP Tour (2000-2024).")
else:
    st.warning("⚠️ Dataset not found. Please check the data path.")
