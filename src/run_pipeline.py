from src.preprocessing import TennisPreprocessor
import pandas as pd
import os

def main():
    # Initialize
    processor = TennisPreprocessor()

    # Load
    print("Loading data...")
    processor.load_data('atp_tennis.csv')

    # Process
    print("Processing data...")
    df = processor.process()

    # Check for NaNs in critical columns
    print("\nChecking for NaNs in engineered features:")
    cols_to_check = ['elo_p1', 'elo_p2', 'p1_win_rate_last_10', 'p1_win_rate_surface']
    print(df[cols_to_check].isna().sum())

    # Split
    print("\nSplitting data...")
    train_df, test_df = processor.time_based_split(test_start_date='2024-01-01')

    if train_df is not None:
        print(f"Train set: {train_df.shape}")
        print(f"Test set: {test_df.shape}")

        # Save
        os.makedirs('data/processed', exist_ok=True)
        train_df.to_csv('data/processed/train_processed.csv', index=False)
        test_df.to_csv('data/processed/test_processed.csv', index=False)
        print("Saved processed data to data/processed/")

        # Verify leakage (simple check)
        # Ensure no test dates in train
        max_train_date = train_df['Date'].max()
        min_test_date = test_df['Date'].min()
        print(f"Max Train Date: {max_train_date}")
        print(f"Min Test Date: {min_test_date}")

        if max_train_date >= min_test_date:
            print("WARNING: LEAKAGE DETECTED! Train dates overlap with Test dates.")
        else:
            print("Split looks correct (chronological).")

if __name__ == "__main__":
    main()
