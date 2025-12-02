import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from src.dashboard.config import DATA_PATH, RANDOM_SEED, TEST_SIZE

@st.cache_data
def load_data():
    """Load and preprocess tennis dataset."""
    if not DATA_PATH.exists():
        st.error(f"Data file not found at {DATA_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # Date conversion
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Filter as per training script
        df = df[df['Date'] > '2010-01-01']

    return df

@st.cache_data
def get_processed_data():
    """Load and preprocess data to include V2 features (Elo, Form, etc.)."""
    processor = TennisPreprocessor()
    processor.load_data(str(DATA_PATH))
    df = processor.process()
    return df

from src.preprocessing import TennisPreprocessor

@st.cache_data
def get_train_test_data():
    """
    Load data and split into train/test sets using the V2 pipeline logic.
    Returns:
        X_train, X_test, y_train, y_test, df_test (original dataframe rows for test set)
    """
    # Use the V2 Preprocessor
    processor = TennisPreprocessor()
    # We need to pass the absolute path to load_data
    processor.load_data(str(DATA_PATH))
    df = processor.process()

    if df is None or df.empty:
        return None, None, None, None, None

    # Split using the same date as training
    # Note: TennisPreprocessor.time_based_split returns full dataframes
    train_df, test_df = processor.time_based_split(test_start_date='2024-01-01')

    if train_df is None or test_df is None:
        return None, None, None, None, None

    # Define all possible features (V1 + V2)
    # This ensures both model types can find their columns
    # V1 models might ignore extra columns if they use a ColumnTransformer that selects specific cols
    # But wait, V1 models were trained on specific columns.
    # If we pass extra columns to a Pipeline with ColumnTransformer, it usually drops the rest.
    # So it is safe to pass all columns.

    # However, we need to separate X and y
    y_train = train_df['y']
    y_test = test_df['y']

    # Drop target and other non-feature columns for X
    # Actually, we can just pass the whole dataframe to the pipeline if the pipeline selects columns.
    # But usually sklearn expects X to be features only.
    # Let's drop 'y', 'Winner', 'Date' etc if needed, or just keep everything.
    # The V2 pipeline selects columns by name.

    X_train = train_df.drop(columns=['y', 'Winner'], errors='ignore')
    X_test = test_df.drop(columns=['y', 'Winner'], errors='ignore')

    return X_train, X_test, y_train, y_test, test_df

def get_player_stats(player_name, df):
    """Get the most recent stats for a player."""
    # Search in Player_1
    p1_matches = df[df['Player_1'] == player_name].sort_values('Date', ascending=False)
    if not p1_matches.empty:
        latest = p1_matches.iloc[0]
        return {
            'rank': int(latest['Rank_1']) if not pd.isna(latest['Rank_1']) else 100,
            'pts': int(latest['Pts_1']) if not pd.isna(latest['Pts_1']) else 1000
        }

    # Search in Player_2
    p2_matches = df[df['Player_2'] == player_name].sort_values('Date', ascending=False)
    if not p2_matches.empty:
        latest = p2_matches.iloc[0]
        return {
            'rank': int(latest['Rank_2']) if not pd.isna(latest['Rank_2']) else 100,
            'pts': int(latest['Pts_2']) if not pd.isna(latest['Pts_2']) else 1000
        }

    return {'rank': 100, 'pts': 1000} # Defaults
