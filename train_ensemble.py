import os
from datetime import datetime
import json
import pandas as pd
import numpy as np
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 42

def get_preprocessor(numerical_cols, categorical_cols):
    """Create the preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

def train_and_export_all():
    """Train multiple models and export them."""
    repo_root = os.path.dirname(__file__)
    csv_path = os.path.join(repo_root, 'atp_tennis.csv')
    models_dir = os.path.join(repo_root, 'models')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    print("Loading data...")
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'] > '2010-01-01']

    y = (df['Winner'] == df['Player_1']).astype(int)

    numerical_cols = ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Best of']
    categorical_cols = ['Series', 'Court', 'Surface', 'Round']

    X = df[numerical_cols + categorical_cols].copy()
    X[numerical_cols] = X[numerical_cols].replace(-1, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    models_config = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=10),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100, max_depth=15),
        'XGBoost': XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
    }

    os.makedirs(models_dir, exist_ok=True)

    for name, clf_model in models_config.items():
        print(f"Training {name}...")

        pipeline = Pipeline(steps=[
            ('preprocessor', get_preprocessor(numerical_cols, categorical_cols)),
            ('classifier', clf_model)
        ])

        pipeline.fit(X_train, y_train)

        # Export
        model_path = os.path.join(models_dir, f"{name}.pkl")
        dump(pipeline, model_path)

        # Metadata
        meta = {
            'exported_at': datetime.utcnow().isoformat() + 'Z',
            'model_path': model_path,
            'seed': RANDOM_SEED,
            'classifier': name,
            'features': numerical_cols + categorical_cols
        }
        meta_path = model_path.replace('.pkl', '.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Exported {name} to {model_path}")

if __name__ == '__main__':
    train_and_export_all()
