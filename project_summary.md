# Project Summary: Tennis Analytics & Learning Dashboard

## 1. Project Overview
This project is a comprehensive educational platform designed to teach **Machine Learning** and **DevOps** concepts through the lens of Tennis Analytics. It combines interactive dashboards, real-world data, and gamified learning modules to bridge the gap between theory and practice.

The application is built using **Streamlit** for the frontend dashboard and likely includes a backend service for model inference.

## 2. Machine Learning Curriculum (Pages 01-10)
This section covers the end-to-end ML pipeline, from data preparation to model evaluation.

### **01. Data & Feature Engineering**
*   **Focus**: Preparing raw data for machine learning.
*   **Key Sections**:
    *   **Feature Engineering**: Techniques to create new features from existing data.

### **02. Model Playground**
*   **Focus**: Interactive sandbox to experiment with different models and datasets.

### **03. Logistic Regression: The Foundation**
*   **Focus**: Deep dive into the most fundamental classification algorithm.
*   **Key Sections**:
    *   **Core Model Definition**: The probabilistic linear classifier.
    *   **Geometry**: Understanding the linear decision boundary.
    *   **Mathematical Intuition**: From Odds to Log-Odds to Sigmoid (Derivation).
    *   **Optimization Problem**: Maximum Likelihood Estimation and Log Loss.
    *   **Gradient Update**: How the model learns via Gradient Descent.
    *   **Interpretation**: Understanding weights as Log-Odds.
    *   **Visualization**: Interactive decision boundary plotting.
    *   **Python Implementation**: `sklearn` code examples.
    *   **Hyperparameters**: Regularization (C) and Penalties (L1/L2).

### **04. Decision Trees & Random Forests**
*   **Focus**: Understanding non-linear models and ensemble methods.
*   **Key Sections**:
    *   **Decision Trees**: The "20 Questions" game analogy.
    *   **Random Forests**: The "Wisdom of the Crowd" concept (Bagging).
    *   **Python Implementation**: Code examples.

### **05. SVM & K-Nearest Neighbors**
*   **Focus**: Geometric and distance-based classifiers.
*   **Key Sections**:
    *   **Support Vector Machines (SVM)**: Margins and kernels.
    *   **K-Nearest Neighbors (KNN)**: The "Lazy Learner" and distance metrics.

### **06. Optimization**
*   **Focus**: How models actually "learn" parameters.
*   **Key Sections**:
    *   **Intuition**: The "Hiker in the Fog" analogy for Gradient Descent.
    *   **The Math**: Step-by-step walkthrough of the update rules.
    *   **Advanced Optimizers**: Adam, RMSprop, and Momentum.
    *   **Visualization**: 3D Loss Surface exploration.
    *   **Python Implementation**: Using optimizers in code.

### **07. Validation & Metrics**
*   **Focus**: Evaluating model performance beyond simple accuracy.
*   **Key Sections**:
    *   **Core Model Definition**: Confusion Matrix basics.
    *   **Precision-Recall Tradeoff**: Balancing False Positives vs False Negatives.
    *   **Advanced Metrics**: F1-Score, ROC-AUC.
    *   **Visualization**: Interactive Threshold Slider.
    *   **Python Implementation**: `classification_report` and other metrics.

### **08. Hyperparameters & Cross-Validation**
*   **Focus**: Tuning models for generalization.
*   **Key Sections**:
    *   **Core Model Definition**: Parameters vs Hyperparameters.
    *   **Bias-Variance Tradeoff**: Overfitting vs Underfitting.
    *   **Cross-Validation**: The "Exam Analogy" (K-Fold).
    *   **Search Strategies**: Grid Search vs Random Search.
    *   **Visualization**: Validation Curves.

### **09. Class Imbalance**
*   **Focus**: Handling datasets where one class is rare (e.g., Fraud, Rare Diseases).
*   **Key Sections**:
    *   **Core Model Definition**: The "Accuracy Paradox".
    *   **The Solutions**: Resampling (SMOTE, Undersampling) and Class Weights.
    *   **Visualization**: The "Swamped Boundary" effect.
    *   **Python Implementation**: `imbalanced-learn` library.

### **10. Model Comparison**
*   **Focus**: Benchmarking all models against each other.
*   **Key Sections**:
    *   **The Leaderboard**: Ranking models by performance metrics.
    *   **Deep Dive Analysis**: Comparative analysis of strengths and weaknesses.

---

## 3. Computer Science & DevOps Curriculum (Pages 11-18)
This section builds the engineering foundation required to deploy ML models.

### **11. Computer Architecture & The OS**
*   **Focus**: How software interacts with hardware.
*   **Key Sections**:
    *   **The Hardware**: CPU, RAM, Disk (The Kitchen analogy).
    *   **The Kernel**: The OS Manager.
    *   **The Process**: How programs run.
    *   **System Calls**: Interface between User Space and Kernel Space.
    *   **Exercises**: Practical tasks.

### **12. Docker Basics**
*   **Focus**: Containerization fundamentals.
*   **Key Sections**:
    *   **The Golden Triangle**: Dockerfile, Image, Container.
    *   **The Dockerfile**: Writing the recipe.
    *   **The Image**: Layers and caching (The Onion).
    *   **The Container**: Runtime instances.
    *   **The Toolbelt**: Essential CLI commands.
    *   **Simulation**: Interactive build process.
    *   **Exercises**: Building and running containers.

### **13. Docker Networking**
*   **Focus**: How containers communicate.
*   **Key Sections**:
    *   **Networking 101**: IP Addresses, Ports, Protocols (Apartment Analogy).
    *   **Docker Networking**: Bridge, Host, None networks.
    *   **Port Mapping**: Exposing containers to the host.
    *   **Common Mistakes**: Localhost confusion.
    *   **Exercises**: Ping tests and port mapping.

### **14. Docker Compose**
*   **Focus**: Multi-container orchestration.
*   **Key Sections**:
    *   **Orchestration 101**: The problem of managing multiple `docker run` commands.
    *   **The Solution**: `docker-compose.yml` (Infrastructure as Code).
    *   **The Lifecycle**: `up`, `down`, `restart`.
    *   **Networking Magic**: Automatic DNS and service discovery.
    *   **Exercises**: Writing a compose file.

### **15. Git Basics**
*   **Focus**: Version control fundamentals.
*   **Key Sections**:
    *   **Git Overview**: Distributed Version Control.
    *   **Installation & Configuration**: Setup.
    *   **The Four Areas**: Working Directory, Staging, Local Repo, Remote Repo.
    *   **Basic Commands**: `add`, `commit`, `status`, `log`.
    *   **Remote Repositories**: GitHub integration.
    *   **Exercises**: Basic workflow.

### **16. Git Collaboration**
*   **Focus**: Working in teams.
*   **Key Sections**:
    *   **The Multiverse**: Branching strategies.
    *   **Merging Strategies**: Fast-forward vs Merge Commit.
    *   **Fetch vs Pull**: Understanding remote synchronization.
    *   **Conflicts**: Resolving merge conflicts.
    *   **Pull Requests & Forking**: Open source workflow.
    *   **Exercises**: Simulating collaboration.

### **17. Git Advanced**
*   **Focus**: Deep dive into Git internals.
*   **Key Sections**:
    *   **The Graph Model (DAG)**: Nodes, parents, and pointers.
    *   **References (Refs)**: Heads, Tags, Remotes.
    *   **Undoing Things**: `reset`, `revert`, `checkout`.
    *   **Reflog**: Recovering "lost" commits.
    *   **Exercises**: Advanced recovery scenarios.

### **18. Docker Advanced**
*   **Focus**: Production-grade Docker concepts.
*   **Key Sections**:
    *   **Docker System**: Daemon, Client, Registry.
    *   **Docker Hub**: Pushing and pulling images.
    *   **Orchestration**: Intro to Swarm/Kubernetes concepts.
    *   **Exercises**: Advanced management.

## 4. Architecture
*   **Frontend**: Streamlit Dashboard (`src/dashboard`).
*   **Backend**: Python Services (`services/`).
*   **Infrastructure**: Docker & Docker Compose (`docker-compose.yml`).
