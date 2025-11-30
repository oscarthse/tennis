import pandas as pd
df = pd.read_csv("atp_tennis.csv")
print(f"Tournament unique: {df['Tournament'].nunique()}")
print(f"Player_1 unique: {df['Player_1'].nunique()}")
print(f"Player_2 unique: {df['Player_2'].nunique()}")
