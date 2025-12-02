# Tennis ML + Docker Learning Guide

This guide explains the project structure, the preprocessing and modeling steps in the notebook, evaluation and baselines, how the service and dashboard work, and Docker/Docker Compose and networking concepts. It's written so you can share this repository with friends to learn end-to-end ML + deployment.

## Project layout (short)

- `oscar_jupyter.ipynb` — the exploratory notebook: data loading, preprocessing, baseline checks, training models (Logistic Regression, Decision Tree, Random Forest, XGBoost), and evaluation. Run top-to-bottom.
- `models/` — place `model.pkl` here (serialized trained model or pipeline).
- `src/service/server.py` — small Flask service that loads `models/model.pkl` and exposes `/health` and `/predict` endpoints.
- `src/dashboard/app.py` — Streamlit dashboard that calls the service and lets you send feature vectors for quick predictions.
- `services/*` — Dockerfiles and `requirements.txt` for service and dashboard images.
- `docker-compose.yml` — composes the `service` and `dashboard` together for local testing.

---

## Notebook: preprocessing and data handling

Key points (your friends should pay attention to these):

- Target creation: `y = (df['Winner'] == df['Player_1']).astype(int)` — target is 1 if Player_1 won.
- Feature selection: numerical and categorical lists defined explicitly (e.g., `Rank_1`, `Rank_2`, `Pts_1`, ... and categorical columns like `Series`, `Court`, `Surface`, `Round`).
- Missing values: some numerical fields use `-1` as a placeholder; replace these with `NaN` so scikit-learn imputers can handle them: `X[numerical_cols] = X[numerical_cols].replace(-1, np.nan)`.
- Preprocessing pipeline:
  - Numerical: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - Categorical: `OneHotEncoder(handle_unknown='ignore')` (produces many binary columns).
  - Combined with `ColumnTransformer` and wrapped in a `Pipeline` so everything is reproducible and safe to persist.

- Important: split the dataset BEFORE fitting the preprocessors (train/test split) to avoid data leakage. The notebook uses `train_test_split(..., stratify=y)` and then fits the preprocessing `pipeline` only on `X_train`.

Why stratify? If the target is imbalanced, stratifying keeps class proportions similar between train and test sets.

---

## Baselines and why they matter

A baseline is a simple prediction strategy you should beat to show your model learned anything meaningful. The notebook includes (or you can add) the following baselines:

- Majority-class baseline: predict the most common class from the training set. Easy to compute; a lower bound.
- Stratified-random baseline: sample classes with the same probability distribution as the train set.
- Uniform random baseline: predict 0/1 with 50/50 probability.
- Favorite-by-odds baseline: if match odds (`Odd_1`, `Odd_2`) are present and valid, predict the lower-odds player as the favorite. Betting odds are often strong predictors; compare your model to this.

Compute these baselines on the same test set as your models so comparisons are fair.

---

## Models in the notebook — short primer

1) Logistic Regression
  - Linear model. Fast, interpretable coefficients, works well when signal is linear.
  - Recommended hyperparameters: `C` (inverse regularization), `penalty` (l2). Use `max_iter` if you see convergence warnings.

2) Decision Tree
  - Non-linear, interpretable via tree plots. Prone to overfitting unless limited (`max_depth`, `min_samples_leaf`).

3) Random Forest
  - Ensemble of decision trees. More robust and usually higher accuracy than single trees. Tune `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`.

4) XGBoost (Extreme Gradient Boosting)
  - Gradient-boosted trees. Powerful, fast, often achieves top performance for tabular data.
  - Tune `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and regularization (`reg_alpha`/`reg_lambda`).

Notes on pipelines
- For `full_pipeline` (the logistic regression example) it's convenient to save the preprocessor + model as a single pipeline and serialize it. The service can then accept raw features and call `pipeline.predict(...)` directly.
- If you train a model on preprocessed DataFrames (`X_train_final`), you must either save the preprocessor or only accept preprocessed features at prediction time.

---

## Evaluation metrics explained (and when to use them)

- Accuracy: fraction of correct predictions. Good general-purpose but misleading for imbalanced classes.
- Precision: TP / (TP + FP). Use when false positives are costly.
- Recall (Sensitivity): TP / (TP + FN). Use when false negatives are costly.
- F1: harmonic mean of precision and recall; single-number summary when class balance between prec/rec matters.
- ROC AUC: area under Receiver Operating Characteristic curve. Measures ranking ability of model scores; uses probability/score outputs rather than hard predictions.

Always compute a few metrics (e.g., accuracy + F1 + ROC AUC) rather than a single one.

Cross-validation
- The notebook uses `StratifiedKFold` to keep class proportions in each fold. Use cross-validation when estimating generalization performance and selecting hyperparameters.

---

## Feature importance and interpretability

- Tree-based models (RandomForest, XGBoost) expose `feature_importances_` which give a global sense of which features matter.
- For more detailed, local + global explanations use SHAP (SHapley Additive exPlanations). SHAP can explain individual predictions and aggregate importances. If you add SHAP, include instructions to `pip install shap` in requirements and show a short example.

---

## Saving models (for the service)

- Use `joblib.dump(pipeline_or_model, 'models/model.pkl')` to serialize.
- Save the entire pipeline if you want the service to accept raw input. If you only save the estimator (without preprocessor), the service must either preprocess inputs or accept preprocessed arrays.

Example:

```python
from joblib import dump
dump(full_pipeline, 'models/model.pkl')
```

---

## Service (`src/service/server.py`) — what it does and expectations

- Endpoint: `GET /health` returns whether the model file was found and loaded.
- Endpoint: `POST /predict` expects JSON `{"X": [[...], [...]]}` — one or more feature vectors. Returns `{"predictions": [...]}`.

Important design choices and tradeoffs
- The service is intentionally minimal. For production: add input validation, authentication, rate-limiting, logging, monitoring, and a schema for feature inputs (JSON Schema).
- Decide whether the service should contain the preprocessor. The easiest UX for your friends: save a full `Pipeline` into `models/model.pkl` so `server.py` can accept raw features.

---

## Dashboard (`src/dashboard/app.py`)

- Streamlit-based quick UI that lets you send CSV-like raw feature rows to the `service` and shows predictions.
- Useful for demos — not intended as a production-grade dashboard but great for learning.

---

## Docker, Dockerfiles, docker-compose, and networking (learning section)

This project includes two Docker images: the `service` and the `dashboard`. `docker-compose.yml` runs both together and sets up networking automatically.

Key concepts to explain to friends:

- Dockerfile: a recipe to build an image. Each service has its own Dockerfile that installs dependencies and copies the app code.
- Image: a read-only snapshot built from a Dockerfile.
- Container: a running instance of an image.
- docker-compose: a convenient way to orchestrate multiple containers locally (services, networks, volumes). `docker-compose.yml` in this repo:
  - builds `services/service` and `services/dashboard` images,
  - maps ports (`5000` for Flask, `8501` for Streamlit),
  - mounts `./models` into the `service` (so you can drop a `model.pkl` locally and the service reads it),
  - by default sets up a user-defined bridge network so services can talk by name: the dashboard can call `http://service:5000/predict` because Docker Compose adds a DNS entry.

Networking notes:

- Containers on the same Compose project share a network; they can resolve each other by service name.
- Port mapping exposes container ports to the host (for browser access). For example, `8501:8501` exposes the dashboard to `http://localhost:8501`.

Development workflow with Docker Compose:

1. Build and run both services:
   ```bash
   docker-compose up --build
   ```
2. Check the dashboard at `http://localhost:8501` and the service at `http://localhost:5000/health`.
3. To update code, you can either rebuild or mount local source into containers (the current Dockerfile copies sources — for live-editing you'd mount volumes in the Compose file).

Security note: never ship secrets in images; use environment variables or Docker secrets for production.

---

## Suggested learning exercises for friends

1. Run the notebook end-to-end and export `models/model.pkl`. Put it in `models/` and run `docker-compose up --build`. Open the dashboard and make predictions.
2. Modify the notebook to train a full pipeline (preprocessor + classifier) and re-export it. Verify the service accepts raw inputs.
3. Add a `GridSearchCV` cell to tune XGBoost hyperparameters and observe how CV scores change.
4. Add SHAP explanations and show a few example game predictions with SHAP force plots.
5. Experiment with Docker: change Compose to mount source directories as volumes and observe code hot-reload.

---

## Next steps and production considerations

- Add input validation (pydantic or marshmallow) to the service.
- Add unit tests for the preprocessing and for the API.
- Add CI to build and test images.
- Add model versioning: store `model-YYYYMMDD.pkl` and serve a specific version or add a model registry.
- Add logging + monitoring + health checks for production readiness.

---

If you'd like, I can:

- Add a `train_and_export.py` script that loads the notebook pipeline variables (or rebuilds the pipeline), trains a final pipeline, and writes `models/model.pkl`.
- Add more learning content (slides or a shorter TL;DR cheat sheet) into `LEARNING_GUIDE_TLDR.md`.

Tell me which follow-up you'd like and I will add it to the repo.
