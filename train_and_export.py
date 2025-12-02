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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def train_and_export(model_path=None):
    """Train a full preprocessing+LogisticRegression pipeline and export to model_path.

    Args:
        model_path (str): path to write the model .pkl file. Defaults to ./models/model.pkl
    """
    repo_root = os.path.dirname(__file__)
    if model_path is None:
        model_path = os.path.join(repo_root, 'models', 'model.pkl')

    csv_path = os.path.join(repo_root, 'atp_tennis.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # minimal filtering to match notebook
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'] > '2010-01-01']

    # target
    y = (df['Winner'] == df['Player_1']).astype(int)

    numerical_cols = ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Best of']
    categorical_cols = ['Series', 'Court', 'Surface', 'Round']

    X = df[numerical_cols + categorical_cols].copy()
    X[numerical_cols] = X[numerical_cols].replace(-1, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # OneHotEncoder parameter name changed across sklearn versions:
    # - older versions use `sparse=`
    # - newer versions use `sparse_output=`
    import inspect
    sig = inspect.signature(OneHotEncoder)
    if 'sparse_output' in sig.parameters:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
    ])

    print('Training pipeline on data...')
    clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(clf, model_path)
    print(f'Model exported to: {model_path}')

    # metadata
    meta = {
        'exported_at': datetime.utcnow().isoformat() + 'Z',
        'model_path': model_path,
        'seed': RANDOM_SEED,
        'classifier': 'LogisticRegression',
    }
    meta_path = model_path.replace('.pkl', '.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Metadata written to: {meta_path}')

    return model_path


if __name__ == '__main__':
    train_and_export()
