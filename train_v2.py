import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import TennisPreprocessor

# Configuration
RANDOM_SEED = 42
TEST_START_DATE = '2024-01-01'

def train_v2_models():
    print("Initializing V2 Training Pipeline...")

    # 1. Preprocessing (Tennis Logic)
    processor = TennisPreprocessor()
    processor.load_data('atp_tennis.csv')
    df = processor.process()

    # 2. Split (Chronological)
    train_df, test_df = processor.time_based_split(test_start_date=TEST_START_DATE)

    if train_df is None or test_df is None:
        print("Error: Split failed.")
        return

    # 3. Define Features
    # We use the new engineered features + some original ones
    numerical_cols = [
        # Elo
        'elo_p1', 'elo_p2', 'elo_prob_p1', 'elo_prob_p2',
        'elo_surf_p1', 'elo_surf_p2',
        # Form / History
        'p1_win_rate_career', 'p1_win_rate_last_10', 'p1_win_rate_surface',
        'p2_win_rate_career', 'p2_win_rate_last_10', 'p2_win_rate_surface',
        # Raw Stats (still useful)
        'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Best of'
    ]

    categorical_cols = ['Series', 'Court', 'Surface', 'Round']

    # Prepare X and y
    X_train = train_df[numerical_cols + categorical_cols]
    y_train = train_df['y']

    X_test = test_df[numerical_cols + categorical_cols]
    y_test = test_df['y']

    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # 4. Model Pipeline (Machine Learning Logic)
    # We still need to encode categoricals and scale numericals

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle any remaining NaNs
        ('scaler', StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Define Models to Train
    models = {
        'LogisticRegression_v2': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        'DecisionTree_v2': DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=10),
        'RandomForest_v2': RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100, max_depth=15),
        'XGBoost_v2': XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }

    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}...")

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        clf.fit(X_train, y_train)

        # Evaluate
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print(f"{name} Results - Train Acc: {train_score:.4f}, Test Acc: {test_score:.4f}")

        # Save
        model_path = os.path.join(models_dir, f"{name}.pkl")
        dump(clf, model_path)

        # Metadata
        meta = {
            'exported_at': datetime.utcnow().isoformat() + 'Z',
            'model_path': model_path,
            'seed': RANDOM_SEED,
            'classifier': name,
            'version': 'v2',
            'test_acc': test_score,
            'features': numerical_cols + categorical_cols,
            'test_start_date': TEST_START_DATE
        }

        meta_path = model_path.replace('.pkl', '.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {name} to {model_path}")

    # 5. Export Player State for Dashboard
    print("Exporting Player State for Dashboard...")
    player_state = {}

    # Get latest Elo from the system
    # Note: processor.elo_system has the state AFTER processing all matches
    elo_system = processor.elo_system

    # Get latest form from dataframe
    # We need the last row for each player to get their latest rolling stats
    # But wait, the features in DF are PRE-match.
    # So for the NEXT match, we need the stats resulting from the LAST match.
    # This is tricky.
    # For Elo, elo_system.ratings IS the post-match rating (ready for next match). Correct.
    # For Rolling stats (win_rate_last_10), we need to calculate it including the last match.

    # Let's re-calculate the "latest" form for every player
    # Iterate over all unique players
    all_players = set(df['Player_1'].unique()) | set(df['Player_2'].unique())

    for player in all_players:
        # Get global Elo
        elo_global = elo_system.ratings.get(player, 1500)

        # Get surface Elos
        elo_surfaces = {}
        for surf, ratings in elo_system.surface_ratings.items():
            if player in ratings:
                elo_surfaces[surf] = ratings[player]

        # Get Form (Win Rate Last 10)
        # Find matches involving this player
        p_matches = df[(df['Player_1'] == player) | (df['Player_2'] == player)].sort_values('Date')

        if not p_matches.empty:
            # Calculate wins
            wins = []
            for _, row in p_matches.iterrows():
                if row['Winner'] == player:
                    wins.append(1)
                else:
                    wins.append(0)

            # Last 10
            last_10 = wins[-10:]
            win_rate_last_10 = sum(last_10) / len(last_10)

            # Career
            win_rate_career = sum(wins) / len(wins)

            # Surface Win Rates
            surface_wins = {}
            surface_counts = {}

            for idx, row in p_matches.iterrows():
                s = row['Surface']
                w = 1 if row['Winner'] == player else 0
                surface_wins[s] = surface_wins.get(s, 0) + w
                surface_counts[s] = surface_counts.get(s, 0) + 1

            win_rate_surfaces = {s: surface_wins[s]/surface_counts[s] for s in surface_wins}

        else:
            win_rate_last_10 = 0.0
            win_rate_career = 0.0
            win_rate_surfaces = {}

        player_state[player] = {
            'elo': elo_global,
            'elo_surfaces': elo_surfaces,
            'win_rate_last_10': win_rate_last_10,
            'win_rate_career': win_rate_career,
            'win_rate_surfaces': win_rate_surfaces
        }

    state_path = os.path.join(models_dir, 'player_state.json')
    with open(state_path, 'w') as f:
        json.dump(player_state, f, indent=2)
    print(f"Saved player state to {state_path}")

if __name__ == "__main__":
    train_v2_models()
