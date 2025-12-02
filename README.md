# 🎾 Tennis ML Platform: From Intuition to Production

**An interactive, open-source learning platform that teaches Machine Learning and Software Engineering using real ATP Tennis data.**

---

## 📖 About The Project

Most tutorials teach Machine Learning on boring datasets like "Iris Flowers" or "Titanic Survivors".
**This project is different.**

We built a full-stack application to predict the winner of professional tennis matches. But instead of just showing you the final code, we built an **Interactive Course** inside the app itself.

**You will learn:**
1.  **Machine Learning**: From the math of Logistic Regression to the complexity of Random Forests and SVMs.
2.  **Data Science**: Feature engineering, handling class imbalance, and rigorous validation.
3.  **Software Engineering**: How computers actually work (Kernel/User Space), Docker containerization, and Git collaboration.

Everything is explained with **Intuition first**, then **Math**, then **Code**, and finally **Interactive Visualization**.

---

## 📚 The Syllabus

The application is divided into interactive modules:

### 🧠 Part 1: Machine Learning
*   **1. Data & Features**: Understanding the ATP dataset, Elo ratings, and "Form".
*   **2. Logistic Regression**: The foundation of probability. Sigmoids and Log-Loss.
*   **3. Trees & Forests**: Decision Trees, Entropy, and how Random Forests fix overfitting.
*   **4. SVM & KNN**: Geometry-based classification. Margins and Neighbors.
*   **5. Optimization**: How Gradient Descent actually works (sliding down the hill).
*   **6. Metrics**: Why Accuracy is dangerous. Precision, Recall, F1, and ROC Curves.
*   **7. Hyperparameters**: Grid Search and Cross-Validation.
*   **8. Class Imbalance**: Handling datasets where one outcome is rare.
*   **9. Model Comparison**: Benchmarking all models on the 2024 Test Set.

### 💻 Part 2: Software Development & DevOps
*   **1. Computer Architecture**: CPU, RAM, Kernel, System Calls, and Processes.
*   **2. Docker Basics**: Images, Containers, Layers, and the "Matrix" analogy.
*   **3. Docker Networking**: Bridges, NAT, DNS, and Packet Flow.
*   **4. Docker Compose**: Orchestration and multi-container architectures.
*   **5. Git Basics**: The DAG, Commits, Staging Area, and Time Travel.
*   **6. Git Collaboration**: Branches, Merging, Conflicts, and Pull Requests.

---

## 🎮 Features

*   **Model Playground**: A sandbox where you can pit players (e.g., Nadal vs. Federer) against each other, tweak their stats (Rank, Odds, Form), and see real-time predictions from trained models.
*   **Interactive Plots**: Visualize Decision Boundaries, ROC Curves, and Calibration Plots using Plotly.
*   **Real Data**: Trained on 20+ years of ATP Tennis matches (2000-2024).

---

## 🚀 Getting Started

You can run this project in three ways:

### Option 1: Streamlit Cloud (Easiest)
[Click here to view the live app](https://learn-ai-software.streamlit.app/)


### Option 2: Docker (Recommended for Devs)
If you have Docker installed, this is the cleanest way to run the app.

```bash
# 1. Clone the repo
git clone https://github.com/ClementStand/tennis.git
cd tennis

# 2. Run with Compose
docker compose up --build
```
The app will be available at `http://localhost:8501`.

### Option 3: Local Python
If you prefer running it directly on your machine:

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run src/dashboard/app.py
```

---

## 🛠️ Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) (Python-based UI)
*   **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
*   **Visualization**: [Plotly](https://plotly.com/)
*   **Infrastructure**: [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/)

---

## 📂 Project Structure

```text
tennis/
├── data/                   # Raw and processed CSV files
├── models/                 # Trained .pkl models and metadata
├── src/
│   ├── dashboard/          # The Streamlit Application
│   │   ├── app.py          # Main entry point
│   │   ├── pages/          # Individual learning modules
│   │   ├── components/     # Reusable UI widgets (Plots, Nav)
│   │   └── utils/          # Helper functions
│   └── preprocessing.py    # Data cleaning pipeline
├── docker-compose.yml      # Container orchestration
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## 🤝 Contributing

This is an open-source project! We welcome contributions.
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
