import pandas as pd
import numpy as np
from datetime import datetime

class EloSystem:
    def __init__(self, k_factor=20, initial_rating=1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}  # Global ratings: {player_name: rating}
        self.surface_ratings = {} # {surface: {player_name: rating}}

    def get_rating(self, player, surface=None):
        """Get current rating for a player. Returns initial_rating if new."""
        # Global
        global_r = self.ratings.get(player, self.initial_rating)

        # Surface specific (if requested)
        if surface:
            if surface not in self.surface_ratings:
                self.surface_ratings[surface] = {}
            surface_r = self.surface_ratings[surface].get(player, self.initial_rating)
            # Weighted average or just return both?
            # For feature engineering, we usually want both separately.
            return global_r, surface_r

        return global_r

    def update_rating(self, winner, loser, surface=None):
        """Update ratings after a match."""
        # Get current ratings
        w_global = self.ratings.get(winner, self.initial_rating)
        l_global = self.ratings.get(loser, self.initial_rating)

        # Calculate expected score
        # E_a = 1 / (1 + 10 ^ ((Rb - Ra) / 400))
        exp_w = 1 / (1 + 10 ** ((l_global - w_global) / 400))
        exp_l = 1 / (1 + 10 ** ((w_global - l_global) / 400))

        # Update global
        # K might be dynamic in future, fixed for now
        new_w = w_global + self.k_factor * (1 - exp_w)
        new_l = l_global + self.k_factor * (0 - exp_l)

        self.ratings[winner] = new_w
        self.ratings[loser] = new_l

        # Update surface specific
        if surface:
            if surface not in self.surface_ratings:
                self.surface_ratings[surface] = {}

            w_surf = self.surface_ratings[surface].get(winner, self.initial_rating)
            l_surf = self.surface_ratings[surface].get(loser, self.initial_rating)

            exp_w_surf = 1 / (1 + 10 ** ((l_surf - w_surf) / 400))
            exp_l_surf = 1 / (1 + 10 ** ((w_surf - l_surf) / 400))

            self.surface_ratings[surface][winner] = w_surf + self.k_factor * (1 - exp_w_surf)
            self.surface_ratings[surface][loser] = l_surf + self.k_factor * (0 - exp_l_surf)

class TennisPreprocessor:
    def __init__(self):
        self.raw_df = None
        self.df = None
        self.elo_system = EloSystem()

    def load_data(self, path):
        """Load raw data and perform initial type conversions."""
        print(f"Loading data from {path}...")
        self.raw_df = pd.read_csv(path)

        # Date conversion
        if 'Date' in self.raw_df.columns:
            self.raw_df['Date'] = pd.to_datetime(self.raw_df['Date'], errors='coerce')

        # Sort chronologically - CRITICAL for time-series
        self.raw_df = self.raw_df.sort_values('Date').reset_index(drop=True)

        print(f"Loaded {len(self.raw_df)} matches.")
        return self.raw_df

    def clean_data(self):
        """
        Clean the dataset:
        - Handle missing values (Rank, Points, Odds).
        - Parse scores (optional, for future feature engineering).
        - Standardize categorical values.
        """
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.raw_df.copy()

        # 1. Filter invalid dates (if any)
        df = df.dropna(subset=['Date'])

        # 2. Impute Ranks and Points
        # Missing rank usually means unranked or very low rank.
        # We impute with a low value (e.g. 2000) rather than median/mean to avoid making them look stronger.
        fill_values = {
            'Rank_1': 2000,
            'Rank_2': 2000,
            'Pts_1': 0,
            'Pts_2': 0,
            'Odd_1': 1.0, # Implied probability ~100% (sure thing) or ~0%?
                          # Actually 1.0 odd means no profit.
                          # Better to flag as missing or impute with 1.0 (neutral/no info) if we use implied prob.
            'Odd_2': 1.0
        }

        for col, val in fill_values.items():
            if col in df.columns:
                # Replace -1 (common placeholder) and NaN
                df[col] = df[col].replace(-1, np.nan).fillna(val)

        # 3. Standardize Surface
        # Check for variations if necessary (e.g. 'Carpet' -> 'Carpet', 'Indoor Hard' -> 'Hard'?)
        # For now, we assume the dataset is relatively clean based on previous inspection,
        # but we can force title case.
        if 'Surface' in df.columns:
            df['Surface'] = df['Surface'].str.title()

        self.df = df
        print("Data cleaning completed.")
        return self.df

    def parse_scores(self):
        """
        Parse the 'Score' column to extract:
        - Total sets played
        - Total games played
        - Retirement flag
        """
        # TODO: Implement complex score parsing if needed for "Dominance" features
        pass

    def add_elo_features(self):
        """
        Calculate and add Elo ratings (Global and Surface) for each match.
        CRITICAL: Must be done chronologically to avoid leakage.
        """
        if self.df is None:
            return

        print("Calculating Elo ratings...")

        # Columns to store ratings
        elo_cols = ['elo_p1', 'elo_p2', 'elo_prob_p1', 'elo_prob_p2',
                    'elo_surf_p1', 'elo_surf_p2']

        # Initialize columns with NaNs
        for col in elo_cols:
            self.df[col] = np.nan

        # Iterate through matches
        # Note: iterrows is slow, but necessary for sequential state updates.
        # For 50k matches it should take a few seconds.

        # We need to map the dataframe index to update it efficiently
        # Or build lists and assign at the end

        elo_p1_list = []
        elo_p2_list = []
        elo_surf_p1_list = []
        elo_surf_p2_list = []
        probs_p1 = []
        probs_p2 = []

        for idx, row in self.df.iterrows():
            p1 = row['Player_1']
            p2 = row['Player_2']
            winner = row['Winner']
            surface = row['Surface']

            # 1. GET ratings BEFORE the match (Features)
            p1_g, p1_s = self.elo_system.get_rating(p1, surface)
            p2_g, p2_s = self.elo_system.get_rating(p2, surface)

            elo_p1_list.append(p1_g)
            elo_p2_list.append(p2_g)
            elo_surf_p1_list.append(p1_s)
            elo_surf_p2_list.append(p2_s)

            # Calculate win probability based on Global Elo
            prob_p1 = 1 / (1 + 10 ** ((p2_g - p1_g) / 400))
            probs_p1.append(prob_p1)
            probs_p2.append(1 - prob_p1)

            # 2. UPDATE ratings AFTER the match (State)
            # Determine winner/loser for update
            if winner == p1:
                self.elo_system.update_rating(p1, p2, surface)
            else:
                self.elo_system.update_rating(p2, p1, surface)

        # Assign back to DataFrame
        self.df['elo_p1'] = elo_p1_list
        self.df['elo_p2'] = elo_p2_list
        self.df['elo_surf_p1'] = elo_surf_p1_list
        self.df['elo_surf_p2'] = elo_surf_p2_list
        self.df['elo_prob_p1'] = probs_p1
        self.df['elo_prob_p2'] = probs_p2

        print("Elo ratings calculated.")
        return self.df

    def add_features(self):
        """
        Add derived features:
        - Recent Form (Win % last 10 matches)
        - Surface Win %
        """
        if self.df is None:
            return

        print("Engineering features...")
        df = self.df.copy()

        # We need a long-format dataframe for player stats (one row per player per match)
        # Player 1 perspective
        p1_df = df[['Date', 'Player_1', 'Winner', 'Surface']].rename(columns={'Player_1': 'Player'})
        p1_df['Opponent'] = df['Player_2']
        p1_df['Won'] = (df['Winner'] == df['Player_1']).astype(int)
        p1_df['Match_ID'] = df.index
        p1_df['Is_P1'] = True

        # Player 2 perspective
        p2_df = df[['Date', 'Player_2', 'Winner', 'Surface']].rename(columns={'Player_2': 'Player'})
        p2_df['Opponent'] = df['Player_1']
        p2_df['Won'] = (df['Winner'] == df['Player_2']).astype(int)
        p2_df['Match_ID'] = df.index
        p2_df['Is_P1'] = False

        # Concatenate
        long_df = pd.concat([p1_df, p2_df]).sort_values(['Player', 'Date'])

        # 1. General Win Rate (Cumulative)
        # Shift 1 to exclude current match
        long_df['matches_played'] = long_df.groupby('Player').cumcount()
        long_df['wins_cumulative'] = long_df.groupby('Player')['Won'].cumsum().shift(1).fillna(0)
        long_df['win_rate_career'] = long_df['wins_cumulative'] / long_df['matches_played'].replace(0, 1)

        # 2. Recent Form (Last 10 matches)
        # Rolling window of size 10, shift 1
        long_df['win_rate_last_10'] = long_df.groupby('Player')['Won'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        ).fillna(0)

        # 3. Surface Win Rate
        # Group by Player AND Surface
        long_df['surface_matches'] = long_df.groupby(['Player', 'Surface']).cumcount()
        long_df['surface_wins'] = long_df.groupby(['Player', 'Surface'])['Won'].cumsum().shift(1).fillna(0)
        long_df['win_rate_surface'] = long_df['surface_wins'] / long_df['surface_matches'].replace(0, 1)

        # Merge back to main DF
        # Features to merge
        feats = ['Match_ID', 'win_rate_career', 'win_rate_last_10', 'win_rate_surface']

        # Merge for P1
        # Filter long_df for rows that originated from P1
        p1_feats = long_df[long_df['Is_P1'] == True][feats].copy()
        p1_feats.columns = ['Match_ID', 'p1_win_rate_career', 'p1_win_rate_last_10', 'p1_win_rate_surface']

        df = df.merge(p1_feats, left_index=True, right_on='Match_ID', how='left')

        # Merge for P2
        # Filter long_df for rows that originated from P2
        p2_feats = long_df[long_df['Is_P1'] == False][feats].copy()
        p2_feats.columns = ['Match_ID', 'p2_win_rate_career', 'p2_win_rate_last_10', 'p2_win_rate_surface']

        df = df.merge(p2_feats, on='Match_ID', how='left')

        self.df = df.drop(columns=['Match_ID'], errors='ignore')
        print("Features engineered.")
        return self.df

    def create_target(self):
        """Create target variable y: 1 if Player_1 wins, 0 otherwise."""
        if self.df is None:
            return
        self.df['y'] = (self.df['Winner'] == self.df['Player_1']).astype(int)

    def time_based_split(self, test_start_date='2024-01-01'):
        """
        Split data into train and test sets based on date.
        Train: < test_start_date
        Test: >= test_start_date
        """
        if self.df is None:
            return None, None

        train_mask = self.df['Date'] < test_start_date
        test_mask = self.df['Date'] >= test_start_date

        train_df = self.df[train_mask].copy()
        test_df = self.df[test_mask].copy()

        print(f"Split data: Train ({len(train_df)}), Test ({len(test_df)})")
        return train_df, test_df

    def process(self):
        """Orchestrate the full pipeline."""
        self.clean_data()
        self.add_elo_features()
        self.add_features()
        self.create_target()
        return self.df
